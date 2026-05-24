use std::fs::{self};
use std::os::unix::fs::PermissionsExt;
use std::path::Component::RootDir;
use std::path::{Path, PathBuf};
use std::time::Duration;

use nix::errno::Errno;
use nix::unistd::Pid;

use super::controller::Controller;
use super::controller_type::{
    CONTROLLER_TYPES, ControllerType, PSEUDO_CONTROLLER_TYPES, PseudoControllerType,
};
use super::cpu::{Cpu, V2CpuControllerError, V2CpuStatsError};
use super::cpuset::CpuSet;
#[cfg(feature = "cgroupsv2_devices")]
use super::devices::Devices;
use super::freezer::{Freezer, V2FreezerError};
use super::hugetlb::{HugeTlb, V2HugeTlbControllerError, V2HugeTlbStatsError};
use super::io::{Io, V2IoControllerError, V2IoStatsError};
use super::memory::{Memory, V2MemoryControllerError, V2MemoryStatsError};
use super::pids::Pids;
use super::unified::{Unified, V2UnifiedError};
use super::util::{self, CGROUP_SUBTREE_CONTROL, V2UtilError};
use crate::common::{
    self, AnyCgroupManager, CGROUP_PROCS, CgroupManager, ControllerOpt, FreezerState,
    JoinSafelyError, PathBufExt, WrapIoResult, WrappedIoError,
};
use crate::stats::{PidStatsError, Stats, StatsProvider};

pub const CGROUP_KILL: &str = "cgroup.kill";

#[derive(thiserror::Error, Debug)]
pub enum V2ManagerError {
    #[error("io error: {0}")]
    WrappedIo(#[from] WrappedIoError),
    #[error("while joining paths: {0}")]
    JoinSafely(#[from] JoinSafelyError),
    #[error(transparent)]
    Util(#[from] V2UtilError),

    #[error(transparent)]
    CpuController(#[from] V2CpuControllerError),
    #[error(transparent)]
    CpuSetController(WrappedIoError),
    #[error(transparent)]
    HugeTlbController(#[from] V2HugeTlbControllerError),
    #[error(transparent)]
    IoController(#[from] V2IoControllerError),
    #[error(transparent)]
    MemoryController(#[from] V2MemoryControllerError),
    #[error(transparent)]
    PidsController(WrappedIoError),
    #[error(transparent)]
    UnifiedController(#[from] V2UnifiedError),
    #[error(transparent)]
    FreezerController(#[from] V2FreezerError),
    #[cfg(feature = "cgroupsv2_devices")]
    #[error(transparent)]
    DevicesController(#[from] super::devices::controller::DevicesControllerError),

    #[error(transparent)]
    CpuStats(#[from] V2CpuStatsError),
    #[error(transparent)]
    HugeTlbStats(#[from] V2HugeTlbStatsError),
    #[error(transparent)]
    PidsStats(PidStatsError),
    #[error(transparent)]
    MemoryStats(#[from] V2MemoryStatsError),
    #[error(transparent)]
    IoStats(#[from] V2IoStatsError),
}

/// Represents a management interface for a cgroup located at `{root_path}/{cgroup_path}`
///
/// This struct does not have ownership of the cgroup
pub struct Manager {
    root_path: PathBuf,
    cgroup_path: PathBuf,
    full_path: PathBuf,
}

impl Manager {
    /// Constructs a new cgroup manager with root path being the mount point
    /// of a cgroup v2 fs and cgroup path being a relative path from the root
    pub fn new(root_path: PathBuf, cgroup_path: PathBuf) -> Result<Self, V2ManagerError> {
        let full_path = root_path.join_safely(&cgroup_path)?;

        Ok(Self {
            root_path,
            cgroup_path,
            full_path,
        })
    }

    /// Creates a unified cgroup at `self.full_path` and attaches a process to it
    fn create_unified_cgroup(&self, pid: Pid) -> Result<(), V2ManagerError> {
        let controllers: Vec<String> = util::get_available_controllers(&self.root_path)?
            .iter()
            .map(|c| format!("+{c}"))
            .collect();

        // Note: we intentionally do NOT write controllers to `self.root_path` here.
        // In nested scenarios (running inside a container where the host's root
        // cgroup is owned by host systemd), writing to the root's
        // `cgroup.subtree_control` fails because the file is on a read-only
        // view from our PoV or owned by another manager. Any ancestor up to
        // root must already have the relevant controllers enabled — otherwise
        // our process could not be executing inside that cgroup hierarchy in
        // the first place. We only enable controllers on path components we
        // ourselves create; for path components that pre-existed our process
        // we tolerate per-controller write failures via
        // `is_subtree_control_per_controller_failure`.

        let mut current_path = self.root_path.clone();
        let mut components = self
            .cgroup_path
            .components()
            .filter(|c| c.ne(&RootDir))
            .peekable();
        while let Some(component) = components.next() {
            current_path = current_path.join(component);
            let we_created = if !current_path.exists() {
                fs::create_dir(&current_path).wrap_create_dir(&current_path)?;
                fs::metadata(&current_path)
                    .wrap_other(&current_path)?
                    .permissions()
                    .set_mode(0o755);
                true
            } else {
                false
            };

            // last component cannot have subtree_control enabled due to internal process constraint
            // if this were set, writing to the cgroups.procs file will fail with Erno 16 (device or resource busy)
            if components.peek().is_some() {
                // When `we_created=true`, we own the cgroup and any failure is
                // a real bug. When `we_created=false`, the cgroup predates us
                // (host systemd, outer container runtime) and per-controller
                // failures on its subtree_control are expected and tolerable
                // — see `is_subtree_control_per_controller_failure` for the
                // errno set we silently skip. This matches the behavior of
                // both runc (opencontainers/cgroups fs2/create.go
                // CreateCgroupPath) and crun (containers/crun
                // libcrun/cgroup-utils.c enable_controllers).
                Self::write_controllers(&current_path, &controllers, /*strict=*/ we_created)?;
            }
        }

        common::write_cgroup_file(self.full_path.join(CGROUP_PROCS), pid)?;
        Ok(())
    }

    /// Returns true if the wrapped IO error indicates that a single
    /// `+controller` write to a `cgroup.subtree_control` file should be
    /// silently skipped when the caller does not own the cgroup. Mirrors the
    /// errno allowlist used by crun's `enable_controllers`
    /// (containers/crun libcrun/cgroup-utils.c) and the unconditional
    /// per-controller swallow in runc's `CreateCgroupPath`
    /// (opencontainers/cgroups fs2/create.go).
    ///
    ///   * `EROFS`      — read-only cgroupfs view (`cgroupns=private` inside
    ///     a container whose root cgroup is host-owned).
    ///   * `EACCES`     — DAC owner is another user (e.g. root-owned ancestor
    ///     slice under a rootless user session).
    ///   * `ENOENT`     — the controller is not present in this cgroup's own
    ///     `cgroup.controllers` (e.g. systemd did not delegate `hugetlb` to
    ///     `user@.service`).
    ///   * `EPERM`      — capability missing (similar to `EACCES`, errno
    ///     varies by kernel path).
    ///   * `EOPNOTSUPP` — controller exists but isn't supported in this
    ///     hierarchy/configuration.
    ///   * `EBUSY`      — controller is temporarily contended by another
    ///     manager (transient; the parent manager will resolve).
    fn is_subtree_control_per_controller_failure(err: &WrappedIoError) -> bool {
        matches!(
            err.inner().raw_os_error().map(Errno::from_raw),
            Some(Errno::EROFS)
                | Some(Errno::EACCES)
                | Some(Errno::ENOENT)
                | Some(Errno::EPERM)
                | Some(Errno::EOPNOTSUPP)
                | Some(Errno::EBUSY)
        )
    }

    /// Writes a list of controllers to the `{path}/cgroup.subtree_control`
    /// file, one at a time, so a single unsupported controller doesn't abort
    /// the whole list.
    ///
    /// When `strict=true`, any per-controller failure is returned to the
    /// caller. When `strict=false`, per-controller failures whose errno is in
    /// the tolerated set (see
    /// [`Self::is_subtree_control_per_controller_failure`]) are logged at
    /// debug level and silently skipped. This matches the behavior of both
    /// runc (opencontainers/cgroups fs2/create.go `CreateCgroupPath`) and
    /// crun (containers/crun libcrun/cgroup-utils.c `enable_controllers`):
    /// the kernel will reject controllers an ancestor cgroup doesn't itself
    /// have enabled, and trying to enable a controller on a cgroup managed
    /// by another manager (host systemd, outer container runtime) is normal
    /// and not an error.
    fn write_controllers(
        path: &Path,
        controllers: &[String],
        strict: bool,
    ) -> Result<(), WrappedIoError> {
        for controller in controllers {
            match common::write_cgroup_file_str(path.join(CGROUP_SUBTREE_CONTROL), controller) {
                Ok(()) => {}
                Err(e) if !strict && Self::is_subtree_control_per_controller_failure(&e) => {
                    tracing::debug!(
                        path = ?path,
                        controller = %controller,
                        errno = ?e.inner().raw_os_error(),
                        "skipping unsupported controller on pre-existing ancestor cgroup",
                    );
                }
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    pub fn any(self) -> AnyCgroupManager {
        AnyCgroupManager::V2(self)
    }
}

impl CgroupManager for Manager {
    type Error = V2ManagerError;

    fn add_task(&self, pid: Pid) -> Result<(), Self::Error> {
        if self.full_path.exists() {
            common::write_cgroup_file(self.full_path.join(CGROUP_PROCS), pid)?;
            return Ok(());
        }
        self.create_unified_cgroup(pid)?;
        Ok(())
    }

    fn apply(&self, controller_opt: &ControllerOpt) -> Result<(), Self::Error> {
        for controller in CONTROLLER_TYPES {
            match controller {
                ControllerType::Cpu => Cpu::apply(controller_opt, &self.full_path)?,
                ControllerType::CpuSet => CpuSet::apply(controller_opt, &self.full_path)?,
                ControllerType::HugeTlb => HugeTlb::apply(controller_opt, &self.full_path)?,
                ControllerType::Io => Io::apply(controller_opt, &self.full_path)?,
                ControllerType::Memory => Memory::apply(controller_opt, &self.full_path)?,
                ControllerType::Pids => Pids::apply(controller_opt, &self.full_path)?,
            }
        }

        #[cfg(feature = "cgroupsv2_devices")]
        Devices::apply(controller_opt, &self.full_path)?;

        for pseudoctlr in PSEUDO_CONTROLLER_TYPES {
            if let PseudoControllerType::Unified = pseudoctlr {
                Unified::apply(
                    controller_opt,
                    &self.full_path,
                    util::get_available_controllers(&self.root_path)?,
                )?;
            }
        }

        Ok(())
    }

    fn remove(&self) -> Result<(), Self::Error> {
        if self.full_path.exists() {
            tracing::debug!("remove cgroup {:?}", self.full_path);
            let kill_file = self.full_path.join(CGROUP_KILL);
            if kill_file.exists() {
                fs::write(&kill_file, "1").wrap_write(&kill_file, "1")?;
            } else {
                let procs_path = self.full_path.join(CGROUP_PROCS);
                let procs = fs::read_to_string(&procs_path).wrap_read(&procs_path)?;

                for line in procs.lines() {
                    let pid: i32 = line
                        .parse()
                        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))
                        .wrap_other(&procs_path)?;
                    let _ = nix::sys::signal::kill(Pid::from_raw(pid), nix::sys::signal::SIGKILL);
                }
            }

            common::delete_with_retry(&self.full_path, 4, Duration::from_millis(100))?;
        }

        Ok(())
    }

    fn freeze(&self, state: FreezerState) -> Result<(), Self::Error> {
        let controller_opt = ControllerOpt {
            resources: &Default::default(),
            freezer_state: Some(state),
            oom_score_adj: None,
            disable_oom_killer: false,
        };
        Ok(Freezer::apply(&controller_opt, &self.full_path)?)
    }

    fn stats(&self) -> Result<Stats, Self::Error> {
        let mut stats = Stats::default();

        for subsystem in CONTROLLER_TYPES {
            match subsystem {
                ControllerType::Cpu => stats.cpu = Cpu::stats(&self.full_path)?,
                ControllerType::HugeTlb => stats.hugetlb = HugeTlb::stats(&self.full_path)?,
                ControllerType::Pids => {
                    stats.pids = Pids::stats(&self.full_path).map_err(V2ManagerError::PidsStats)?
                }
                ControllerType::Memory => stats.memory = Memory::stats(&self.full_path)?,
                ControllerType::Io => stats.blkio = Io::stats(&self.full_path)?,
                _ => continue,
            }
        }

        Ok(stats)
    }

    fn get_all_pids(&self) -> Result<Vec<Pid>, Self::Error> {
        Ok(common::get_all_pids(&self.full_path)?)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::test::set_fixture;
    use crate::v2::util::CGROUP_CONTROLLERS;

    /// `is_subtree_control_per_controller_failure` must match the full
    /// crun-parity errno allowlist (EROFS, EACCES, ENOENT, EPERM,
    /// EOPNOTSUPP, EBUSY) and must reject every other errno so we never
    /// silently swallow legitimate write failures.
    #[test]
    fn is_subtree_control_per_controller_failure_matches() {
        fn wrap(errno: Errno) -> WrappedIoError {
            WrappedIoError::Write {
                err: std::io::Error::from_raw_os_error(errno as i32),
                path: PathBuf::from("/some/cgroup/cgroup.subtree_control"),
                data: "+cpu".into(),
            }
        }

        // Tolerated errnos — all must match.
        assert!(Manager::is_subtree_control_per_controller_failure(&wrap(
            Errno::EROFS
        )));
        assert!(Manager::is_subtree_control_per_controller_failure(&wrap(
            Errno::EACCES
        )));
        assert!(Manager::is_subtree_control_per_controller_failure(&wrap(
            Errno::ENOENT
        )));
        assert!(Manager::is_subtree_control_per_controller_failure(&wrap(
            Errno::EPERM
        )));
        assert!(Manager::is_subtree_control_per_controller_failure(&wrap(
            Errno::EOPNOTSUPP
        )));
        assert!(Manager::is_subtree_control_per_controller_failure(&wrap(
            Errno::EBUSY
        )));

        // Untolerated errnos — must NOT match.
        assert!(!Manager::is_subtree_control_per_controller_failure(&wrap(
            Errno::ENOSPC
        )));
        assert!(!Manager::is_subtree_control_per_controller_failure(&wrap(
            Errno::EINVAL
        )));
        assert!(!Manager::is_subtree_control_per_controller_failure(&wrap(
            Errno::EIO
        )));
    }

    /// End-to-end happy-path exercise of `create_unified_cgroup` against a
    /// fully-writable fake cgroupfs in a tempdir. This guards against the
    /// regression where removing the unconditional `write_controllers` on the
    /// root path would have broken nested setups: the old code would have
    /// required `root_path/cgroup.subtree_control` to exist and be writable,
    /// while the new code intentionally skips that write.
    ///
    /// The fake layout is:
    ///     <root>/cgroup.controllers             -> "cpu memory pids"
    ///     <root>/parent/cgroup.subtree_control  -> "" (pre-existing)
    ///     <root>/parent/leaf/                   -> created by create_unified_cgroup
    ///     <root>/parent/leaf/cgroup.procs       -> pre-created so write succeeds
    ///
    /// Note we *do not* create `<root>/cgroup.subtree_control`; the old code
    /// would have aborted with ENOENT trying to write it.
    #[test]
    fn create_unified_cgroup_skips_root_subtree_control_write() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let root = tmp.path();

        // `get_available_controllers` reads this file.
        set_fixture(root, CGROUP_CONTROLLERS, "cpu memory pids").expect("write cgroup.controllers");

        // Pre-existing parent ancestor with a writable subtree_control file.
        let parent = root.join("parent");
        fs::create_dir(&parent).expect("create parent dir");
        set_fixture(&parent, CGROUP_SUBTREE_CONTROL, "").expect("write parent subtree_control");

        // We do *not* pre-create the leaf directory; create_unified_cgroup
        // must mkdir it. However its `cgroup.procs` needs to exist for the
        // final `write_cgroup_file` call to open it (create=false).
        //
        // Pre-creating the file before mkdir is impossible, so instead we
        // wedge open by pre-creating the leaf dir + procs file (which
        // means we exercise the `current_path.exists()` true branch for
        // the leaf — that's fine because the leaf has no subtree_control
        // write gate).
        let leaf = parent.join("leaf");
        fs::create_dir(&leaf).expect("create leaf dir");
        set_fixture(&leaf, CGROUP_PROCS, "").expect("write leaf cgroup.procs");

        let manager = Manager::new(root.to_path_buf(), PathBuf::from("/parent/leaf"))
            .expect("construct manager");

        // Pid 0 is fine for the test; we just need write_cgroup_file to
        // round-trip the bytes into the file.
        manager
            .create_unified_cgroup(Pid::from_raw(0))
            .expect("create_unified_cgroup succeeds when root subtree_control is absent");

        // Sanity: the pid we wrote should be in the procs file.
        let procs = fs::read_to_string(leaf.join(CGROUP_PROCS)).expect("read cgroup.procs");
        assert_eq!(procs.trim(), "0");
    }

    /// `write_controllers` with `strict=true` (we own the cgroup) must
    /// propagate any underlying write failure. `strict=false` (pre-existing
    /// ancestor) must tolerate tolerated errnos. We can't inject specific
    /// errnos through the tempfile-backed write path easily, but we CAN
    /// verify the happy path round-trips in both strict modes against a
    /// writable subtree_control file.
    #[test]
    fn write_controllers_happy_path_both_modes() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let dir = tmp.path();
        set_fixture(dir, CGROUP_SUBTREE_CONTROL, "").expect("write subtree_control fixture");

        let controllers = vec!["+cpu".to_string(), "+memory".to_string()];

        // strict=true succeeds against a writable file.
        Manager::write_controllers(dir, &controllers, /*strict=*/ true)
            .expect("strict write_controllers succeeds on writable subtree_control");

        // strict=false also succeeds against a writable file (happy path
        // tolerance does not regress correctness).
        Manager::write_controllers(dir, &controllers, /*strict=*/ false)
            .expect("non-strict write_controllers succeeds on writable subtree_control");
    }
}
