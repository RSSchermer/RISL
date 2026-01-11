use slir::{Function, Symbol};
use urlencoding::encode as urlencode;

pub fn function_url(module: Symbol, function: Function) -> String {
    format!(
        "/{}/functions/{}--{}",
        urlencode(module.as_str()),
        urlencode(function.module.as_str()),
        urlencode(function.name.as_str())
    )
}
