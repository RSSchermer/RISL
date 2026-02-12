use std::fs::File;

use ar::{GnuBuilder, Header};
use rustc_span::def_id::LocalModDefId;
use slir::cfg::Cfg;
use slir::rvsdg::Rvsdg;
use slir::scf::Scf;
use slir::smi::ShaderModuleInterface;
use slir::{Module, Symbol};

use crate::context::RislContext;

pub const MODULE_IDENTIFIER: &'static str = "module";
pub const CFG_IDENTIFIER: &'static str = "cfg";
pub const SCF_IDENTIFIER: &'static str = "scf";
pub const RVSDG_INITIAL_IDENTIFIER: &'static str = "rvsdg_initial";
pub const RVSDG_TRANSFORMED_IDENTIFIER: &'static str = "rvsdg_transformed";
pub const WGSL_IDENTIFIER: &'static str = "wgsl";
pub const SMI_IDENTIFIER: &'static str = "smi";

pub struct SlirArtifactBuilderConfig {
    pub module_id: LocalModDefId,
    pub include_rvsdg_initial: bool,
    pub include_rvsdg_transformed: bool,
    pub include_wgsl: bool,
    pub include_smi: bool,
}

pub struct SlirArtifactBuilder {
    inner: GnuBuilder<File>,
    module_identifier: Vec<u8>,
    cfg_identifier: Vec<u8>,
    scf_identifier: Vec<u8>,
    rvsdg_initial_identifier: Option<Vec<u8>>,
    rvsdg_transformed_identifier: Option<Vec<u8>>,
    wgsl_identifier: Option<Vec<u8>>,
    smi_identifier: Option<Vec<u8>>,
}

impl SlirArtifactBuilder {
    pub fn new(cx: &RislContext, config: SlirArtifactBuilderConfig) -> Self {
        let SlirArtifactBuilderConfig {
            module_id,
            include_rvsdg_initial,
            include_rvsdg_transformed,
            include_wgsl,
            include_smi,
        } = config;

        let filename = cx.shader_artifact_file_path(module_id.to_local_def_id());
        let file = File::create(filename).expect("failed to create slir artifact file");

        let module_identifier = MODULE_IDENTIFIER.as_bytes().to_vec();
        let cfg_identifier = CFG_IDENTIFIER.as_bytes().to_vec();
        let scf_identifier = SCF_IDENTIFIER.as_bytes().to_vec();
        let rvsdg_initial_identifier =
            include_rvsdg_initial.then(|| RVSDG_INITIAL_IDENTIFIER.as_bytes().to_vec());
        let rvsdg_transformed_identifier =
            include_rvsdg_transformed.then(|| RVSDG_TRANSFORMED_IDENTIFIER.as_bytes().to_vec());
        let wgsl_identifier = include_wgsl.then(|| WGSL_IDENTIFIER.as_bytes().to_vec());
        let smi_identifier = include_smi.then(|| SMI_IDENTIFIER.as_bytes().to_vec());

        let mut identifiers = vec![
            module_identifier.clone(),
            cfg_identifier.clone(),
            scf_identifier.clone(),
        ];

        if let Some(rvsdg_initial_identifier) = &rvsdg_initial_identifier {
            identifiers.push(rvsdg_initial_identifier.clone());
        }

        if let Some(rvsdg_transformed_identifier) = &rvsdg_transformed_identifier {
            identifiers.push(rvsdg_transformed_identifier.clone());
        }

        if let Some(wgsl_identifier) = &wgsl_identifier {
            identifiers.push(wgsl_identifier.clone());
        }

        if let Some(smi_identifier) = &smi_identifier {
            identifiers.push(smi_identifier.clone());
        }

        let inner = GnuBuilder::new(file, identifiers);

        SlirArtifactBuilder {
            inner,
            module_identifier,
            cfg_identifier,
            scf_identifier,
            rvsdg_initial_identifier,
            rvsdg_transformed_identifier,
            wgsl_identifier,
            smi_identifier,
        }
    }

    pub fn add_cfg(&mut self, cfg: &Cfg) {
        let encoding = bincode::serde::encode_to_vec(cfg, bincode::config::standard())
            .expect("failed to encode SLIR Control-Flow Graph");

        self.inner
            .append(
                &Header::new(self.cfg_identifier.clone(), encoding.len() as u64),
                encoding.as_slice(),
            )
            .expect("failed to append SLIR Control-Flow Graph to SLIR artifact archive");
    }

    pub fn add_scf(&mut self, cfg: &Scf) {
        let encoding = bincode::serde::encode_to_vec(cfg, bincode::config::standard())
            .expect("failed to encode SLIR Structured Control-Flow");

        self.inner
            .append(
                &Header::new(self.scf_identifier.clone(), encoding.len() as u64),
                encoding.as_slice(),
            )
            .expect("failed to append SLIR Structured Control-Flow to SLIR artifact archive");
    }

    pub fn maybe_add_rvsdg_initial(&mut self, rvsdg: &Rvsdg) {
        if let Some(identifier) = self.rvsdg_initial_identifier.clone() {
            let encoding = bincode::serde::encode_to_vec(rvsdg.as_data(), bincode::config::standard())
                .expect("failed to encode SLIR RVSDG-initial");

            self.inner
                .append(
                    &Header::new(identifier, encoding.len() as u64),
                    encoding.as_slice(),
                )
                .expect("failed to append SLIR RVSDG-initial to SLIR artifact archive");
        }
    }

    pub fn maybe_add_rvsdg_transformed(&mut self, rvsdg: &Rvsdg) {
        if let Some(identifier) = self.rvsdg_transformed_identifier.clone() {
            let encoding = bincode::serde::encode_to_vec(rvsdg.as_data(), bincode::config::standard())
                .expect("failed to encode SLIR RVSDG-transformed");

            self.inner
                .append(
                    &Header::new(identifier, encoding.len() as u64),
                    encoding.as_slice(),
                )
                .expect("failed to append SLIR RVSDG-transformed to SLIR artifact archive");
        }
    }

    pub fn maybe_add_wgsl(&mut self, wgsl: &str) {
        if let Some(identifier) = self.wgsl_identifier.clone() {
            self.inner
                .append(&Header::new(identifier, wgsl.len() as u64), wgsl.as_bytes())
                .expect("failed to append WGSL to SLIR artifact archive");
        }
    }

    pub fn maybe_add_smi(&mut self, smi: &ShaderModuleInterface) {
        if let Some(identifier) = self.smi_identifier.clone() {
            let encoding = bincode::serde::encode_to_vec(smi, bincode::config::standard())
                .expect("failed to encode SMI");

            self.inner
                .append(
                    &Header::new(identifier, encoding.len() as u64),
                    encoding.as_slice(),
                )
                .expect("failed to append SMI to SLIR artifact archive");
        }
    }

    pub fn finish(mut self, module: &Module) {
        let encoding = bincode::serde::encode_to_vec(&module, bincode::config::standard())
            .expect("failed to encode SLIR module");

        self.inner
            .append(
                &Header::new(self.module_identifier, encoding.len() as u64),
                encoding.as_slice(),
            )
            .expect("failed to append SLIR module to SLIR artifact archive");
    }
}
