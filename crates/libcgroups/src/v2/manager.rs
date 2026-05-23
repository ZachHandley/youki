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
        // `cgroup.subtree_control` fails with EROFS because the file is on a
        // read-only view from our PoV. Any ancestor up to root must already
        // have the relevant controllers enabled — otherwise our process could
        // not be executing inside that cgroup hierarchy in the first place. We
        // only enable controllers on path components we ourselves create.

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
                match Self::write_controllers(&current_path, &controllers) {
                    Ok(()) => {}
                    Err(e) if !we_created && Self::is_erofs(&e) => {
                        // Pre-existing ancestor owned by a parent cgroup
                        // manager (e.g. host systemd, the outer container's
                        // runtime). Controllers are presumed already enabled —
                        // otherwise we could not be running here. Skip.
                        tracing::debug!(
                            path = ?current_path,
                            "skipping subtree_control write on pre-existing read-only ancestor",
                        );
                    }
                    Err(e) => return Err(e.into()),
                }
            }
        }

        common::write_cgroup_file(self.full_path.join(CGROUP_PROCS), pid)?;
        Ok(())
    }

    /// Returns true if the wrapped IO error originates from an EROFS
    /// (read-only file system) syscall failure.
    fn is_erofs(err: &WrappedIoError) -> bool {
        matches!(
            err.inner().raw_os_error().map(Errno::from_raw),
            Some(Errno::EROFS)
        )
    }

    /// Writes a list of controllers to the `{path}/cgroup.subtree_control` file
    fn write_controllers(path: &Path, controllers: &[String]) -> Result<(), WrappedIoError> {
        for controller in controllers {
            common::write_cgroup_file_str(path.join(CGROUP_SUBTREE_CONTROL), controller)?;
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

    /// `is_erofs` correctly identifies EROFS-wrapped IO errors and rejects
    /// every other errno.
    #[test]
    fn is_erofs_recognises_erofs() {
        let erofs = std::io::Error::from_raw_os_error(Errno::EROFS as i32);
        let wrapped = WrappedIoError::Write {
            err: erofs,
            path: PathBuf::from("/some/cgroup/cgroup.subtree_control"),
            data: "+cpu".into(),
        };
        assert!(Manager::is_erofs(&wrapped));

        let eacces = std::io::Error::from_raw_os_error(Errno::EACCES as i32);
        let wrapped = WrappedIoError::Write {
            err: eacces,
            path: PathBuf::from("/some/cgroup/cgroup.subtree_control"),
            data: "+cpu".into(),
        };
        assert!(!Manager::is_erofs(&wrapped));

        let ebusy = WrappedIoError::Open {
            err: std::io::Error::from_raw_os_error(Errno::EBUSY as i32),
            path: PathBuf::from("/some/cgroup/cgroup.subtree_control"),
        };
        assert!(!Manager::is_erofs(&ebusy));
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
}
