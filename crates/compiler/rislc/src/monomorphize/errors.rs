use rustc_macros::{Diagnostic, LintDiagnostic};
use rustc_public::mir::mono::Instance;
use rustc_span::{Span, Symbol};

#[derive(Diagnostic)]
#[diag(rislc_monomorphize_recursion_limit)]
pub(crate) struct RecursionLimit {
    #[primary_span]
    pub span: Span,
    pub instance: String,
    #[note]
    pub def_span: Span,
    pub def_path_str: String,
}

#[derive(Diagnostic)]
#[diag(rislc_monomorphize_encountered_error_while_instantiating)]
pub(crate) struct EncounteredErrorWhileInstantiating {
    #[primary_span]
    pub span: Span,
    pub kind: &'static str,
    pub instance: String,
}
