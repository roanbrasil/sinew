//! `prompt!` and `chat_prompt!`: prompt templates checked at compile time.
//!
//! The template is parsed with the same rules as `sinew_core::PromptTemplate`
//! (`{name}` placeholders, `{{` / `}}` for literal braces). The expansion is a
//! typestate builder: one generic parameter per variable, `Unset` until its
//! setter is called. `format()` requires every parameter to be set, so a
//! missing variable is a type error and an unknown one is a missing method.

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    Ident, LitStr, Path, Token,
};

/// Methods of the generated type; a variable with one of these names would
/// shadow (or be shadowed by) it.
const RESERVED: &[&str] = &[
    "format",
    "to_template",
    "to_vars",
    "to_messages",
    "variables",
];

// ------------------------------------------------------------------ parsing

/// Template text, from a literal or a file read at compile time.
struct Source {
    text: String,
    span: Span,
    /// `(path as written, absolute path)` for `file = "..."`.
    file: Option<(String, String)>,
}

impl Source {
    fn parse_with(input: ParseStream<'_>, macro_name: &str) -> syn::Result<Self> {
        if input.peek(LitStr) {
            let lit: LitStr = input.parse()?;
            return Ok(Self {
                text: lit.value(),
                span: lit.span(),
                file: None,
            });
        }
        if input.peek(Ident) && input.fork().parse::<Ident>()? == "file" {
            input.parse::<Ident>()?;
            input.parse::<Token![=]>()?;
            let lit: LitStr = input.parse()?;
            return read_file(&lit);
        }
        Err(input.error(format!(
            "{macro_name} expects a string literal template or `file = \"path\"`"
        )))
    }
}

fn read_file(lit: &LitStr) -> syn::Result<Source> {
    let rel = lit.value();
    let path = std::path::Path::new(&rel);
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        let dir = std::env::var("CARGO_MANIFEST_DIR").map_err(|_| {
            syn::Error::new(
                lit.span(),
                "CARGO_MANIFEST_DIR is not set; build with cargo or use an absolute path",
            )
        })?;
        std::path::Path::new(&dir).join(path)
    };
    let text = std::fs::read_to_string(&abs).map_err(|e| {
        let why = match e.kind() {
            std::io::ErrorKind::NotFound => "file not found".to_string(),
            _ => e.to_string(),
        };
        syn::Error::new(
            lit.span(),
            format!(
                "cannot read prompt file `{rel}` (paths are relative to the crate's \
                 CARGO_MANIFEST_DIR): {why}"
            ),
        )
    })?;
    Ok(Source {
        text,
        span: lit.span(),
        file: Some((rel, abs.to_string_lossy().into_owned())),
    })
}

/// `crate = "path"` override, shared by both macros.
fn parse_crate(input: ParseStream<'_>) -> syn::Result<Path> {
    input.parse::<Token![crate]>()?;
    input.parse::<Token![=]>()?;
    let lit: LitStr = input.parse()?;
    lit.parse::<Path>()
}

/// `prompt!("...")`, `prompt!(file = "...")`, each with an optional
/// trailing `crate = "..."`.
pub(crate) struct PromptInput {
    source: Source,
    krate: Option<Path>,
}

impl Parse for PromptInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let source = Source::parse_with(input, "prompt!")?;
        let mut krate = None;
        if input.parse::<Option<Token![,]>>()?.is_some() && !input.is_empty() {
            krate = Some(parse_crate(input)?);
            input.parse::<Option<Token![,]>>()?;
        }
        if !input.is_empty() {
            return Err(input.error(
                "unexpected input; prompt! takes one template: `prompt!(\"...\")` or \
                 `prompt!(file = \"path\")`",
            ));
        }
        Ok(Self { source, krate })
    }
}

#[derive(Clone, Copy)]
enum ChatRole {
    System,
    User,
    Assistant,
}

/// `chat_prompt!(system: "...", user: file = "...", assistant: "...")`.
pub(crate) struct ChatInput {
    messages: Vec<(ChatRole, Source)>,
    krate: Option<Path>,
}

impl Parse for ChatInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut messages = Vec::new();
        let mut krate = None;
        while !input.is_empty() {
            if input.peek(Token![crate]) {
                krate = Some(parse_crate(input)?);
            } else {
                let role: Ident = input.parse()?;
                let role = match role.to_string().as_str() {
                    "system" => ChatRole::System,
                    "user" => ChatRole::User,
                    "assistant" => ChatRole::Assistant,
                    other => {
                        return Err(syn::Error::new(
                            role.span(),
                            format!(
                                "unknown role `{other}` in chat_prompt!; expected `system`, \
                                 `user` or `assistant`"
                            ),
                        ))
                    }
                };
                input.parse::<Token![:]>()?;
                messages.push((role, Source::parse_with(input, "chat_prompt!")?));
            }
            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }
        if messages.is_empty() {
            return Err(syn::Error::new(
                Span::call_site(),
                "chat_prompt! needs at least one message, e.g. \
                 `chat_prompt!(system: \"You are {persona}.\", user: \"{question}\")`",
            ));
        }
        Ok(Self { messages, krate })
    }
}

// ----------------------------------------------------------------- template

enum Piece {
    Text(String),
    Var(String),
}

/// Splits a template exactly like `PromptTemplate::pieces` in sinew-core;
/// literal braces are folded into the surrounding text.
fn parse_template(s: &str) -> Result<Vec<Piece>, String> {
    fn text(out: &mut Vec<Piece>, t: &str) {
        if t.is_empty() {
            return;
        }
        match out.last_mut() {
            Some(Piece::Text(prev)) => prev.push_str(t),
            _ => out.push(Piece::Text(t.to_string())),
        }
    }
    let bytes = s.as_bytes();
    let (mut out, mut i, mut start) = (Vec::new(), 0, 0);
    while i < bytes.len() {
        match bytes[i] {
            b'{' if bytes.get(i + 1) == Some(&b'{') => {
                text(&mut out, &s[start..i]);
                text(&mut out, "{");
                i += 2;
                start = i;
            }
            b'}' if bytes.get(i + 1) == Some(&b'}') => {
                text(&mut out, &s[start..i]);
                text(&mut out, "}");
                i += 2;
                start = i;
            }
            b'{' => {
                let end = s[i + 1..].find('}').ok_or_else(|| {
                    format!("unclosed `{{` at byte {i}; write `{{{{` for a literal brace")
                })?;
                let name = s[i + 1..i + 1 + end].trim();
                if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    return Err(format!(
                        "invalid variable name `{name}`; use letters, digits and `_`, \
                         or write `{{{{` / `}}}}` for literal braces"
                    ));
                }
                text(&mut out, &s[start..i]);
                out.push(Piece::Var(name.to_string()));
                i += end + 2;
                start = i;
            }
            b'}' => {
                return Err(format!(
                    "unmatched `}}` at byte {i}; write `}}}}` for a literal brace"
                ))
            }
            _ => i += 1,
        }
    }
    text(&mut out, &s[start..]);
    Ok(out)
}

/// The Rust identifier for a variable's setter: the name itself, or a raw
/// identifier for keywords (`{type}` → `r#type`).
fn var_ident(name: &str) -> Result<Ident, String> {
    if RESERVED.contains(&name) {
        return Err(format!(
            "variable name `{name}` collides with the `{name}()` method of the generated \
             prompt type; rename the variable"
        ));
    }
    syn::parse_str::<Ident>(name)
        .or_else(|_| syn::parse_str::<Ident>(&format!("r#{name}")))
        .map_err(|_| {
            format!(
                "variable name `{name}` is valid for PromptTemplate but is not a Rust \
                 identifier, so prompt! cannot generate a setter for it; rename it or use \
                 PromptTemplate::new at runtime"
            )
        })
}

/// A parsed template ready for code generation.
struct Parsed {
    pieces: Vec<Piece>,
    text: String,
    track: Option<String>,
}

fn parse_source(src: Source, vars: &mut Vec<(String, Ident)>) -> syn::Result<Parsed> {
    let in_file = |msg: String| match &src.file {
        Some((rel, _)) => format!("in prompt file `{rel}`: {msg}"),
        None => format!("invalid prompt template: {msg}"),
    };
    let pieces = parse_template(&src.text).map_err(|m| syn::Error::new(src.span, in_file(m)))?;
    for p in &pieces {
        if let Piece::Var(name) = p {
            if !vars.iter().any(|(n, _)| n == name) {
                let id = var_ident(name).map_err(|m| syn::Error::new(src.span, in_file(m)))?;
                vars.push((name.clone(), id));
            }
        }
    }
    Ok(Parsed {
        pieces,
        text: src.text,
        track: src.file.map(|(_, abs)| abs),
    })
}

// ---------------------------------------------------------------- expansion

pub(crate) fn expand_prompt(input: PromptInput) -> syn::Result<TokenStream2> {
    let krate = match input.krate {
        Some(p) => quote!(#p),
        None => crate::core_path("prompt!")?,
    };
    let mut vars = Vec::new();
    let parsed = parse_source(input.source, &mut vars)?;
    let p = quote!(#krate::__private::prompt);
    let (generics, where_clause) = shape(&vars);
    let render = render_expr(&parsed.pieces, &vars, &p);
    let text = &parsed.text;
    let finish = quote! {
        /// The rendered prompt.
        fn format(&self) -> ::std::string::String #where_clause {
            #render
        }
    };
    let to_template = quote! {
        /// The template as a runtime [`PromptTemplate`].
        fn to_template(&self) -> #krate::PromptTemplate {
            #p::template(#text)
        }
    };
    let from = quote! {
        fn from(p: Prompt #generics) -> Self { p.to_template() }
    };
    Ok(builder(
        &krate,
        &vars,
        format_ident!("Prompt"),
        parsed.track.iter().collect(),
        vec![finish, to_template],
        quote!(#krate::PromptTemplate),
        from,
    ))
}

pub(crate) fn expand_chat(input: ChatInput) -> syn::Result<TokenStream2> {
    let krate = match input.krate {
        Some(p) => quote!(#p),
        None => crate::core_path("chat_prompt!")?,
    };
    let p = quote!(#krate::__private::prompt);
    let mut vars = Vec::new();
    let mut parts = Vec::new();
    for (role, src) in input.messages {
        parts.push((role, parse_source(src, &mut vars)?));
    }
    let (generics, where_clause) = shape(&vars);
    let role_expr = |r: ChatRole| match r {
        ChatRole::System => quote!(#krate::Role::System),
        ChatRole::User => quote!(#krate::Role::User),
        ChatRole::Assistant => quote!(#krate::Role::Assistant),
    };
    let messages: Vec<TokenStream2> = parts
        .iter()
        .map(|(r, parsed)| {
            let role = role_expr(*r);
            let render = render_expr(&parsed.pieces, &vars, &p);
            quote!(#p::message(#role, #render))
        })
        .collect();
    let templates: Vec<TokenStream2> = parts
        .iter()
        .map(|(r, parsed)| {
            let role = role_expr(*r);
            let text = &parsed.text;
            quote!((#role, #text))
        })
        .collect();
    let methods = vec![
        quote! {
            /// The rendered messages, in order.
            fn to_messages(&self) -> ::std::vec::Vec<#krate::Message> #where_clause {
                ::std::vec![#(#messages),*]
            }
        },
        quote! {
            /// A [`ChatRequest`] with the rendered messages.
            fn format(&self) -> #krate::ChatRequest #where_clause {
                #krate::ChatRequest::new(self.to_messages())
            }
        },
        quote! {
            /// The messages as a runtime [`ChatPromptTemplate`].
            fn to_template(&self) -> #krate::ChatPromptTemplate {
                #p::chat_template(&[#(#templates),*])
            }
        },
    ];
    let from = quote! {
        fn from(p: ChatPrompt #generics) -> Self { p.to_template() }
    };
    let tracks: Vec<&String> = parts.iter().filter_map(|(_, x)| x.track.as_ref()).collect();
    Ok(builder(
        &krate,
        &vars,
        format_ident!("ChatPrompt"),
        tracks,
        methods,
        quote!(#krate::ChatPromptTemplate),
        from,
    ))
}

/// A block expression that builds a `String` from the pieces, reading each
/// variable through `Value::get` on its field.
fn render_expr(pieces: &[Piece], vars: &[(String, Ident)], p: &TokenStream2) -> TokenStream2 {
    let literal_len: usize = pieces
        .iter()
        .map(|x| match x {
            Piece::Text(t) => t.len(),
            Piece::Var(_) => 0,
        })
        .sum();
    let pushes = pieces.iter().map(|x| match x {
        Piece::Text(t) => quote!(__out.push_str(#t);),
        Piece::Var(name) => {
            let i = vars.iter().position(|(n, _)| n == name).unwrap_or_default();
            let field = format_ident!("__v{}", i);
            quote!(__out.push_str(#p::Value::get(&self.#field));)
        }
    });
    quote! {{
        let mut __out = ::std::string::String::with_capacity(#literal_len);
        #(#pushes)*
        __out
    }}
}

/// Generic parameters (`<__T0, ..>`) and the `where` clause that requires
/// every variable to be set.
fn shape(vars: &[(String, Ident)]) -> (TokenStream2, TokenStream2) {
    if vars.is_empty() {
        return (quote!(), quote!());
    }
    let params = (0..vars.len()).map(|i| format_ident!("__T{}", i));
    let bounds = vars.iter().enumerate().map(|(i, (v, _))| {
        let t = format_ident!("__T{}", i);
        let tr = format_ident!("Var_{}", v);
        quote!(#t: #tr)
    });
    (quote!(<#(#params),*>), quote!(where #(#bounds),*))
}

/// The typestate builder shared by both macros.
fn builder(
    krate: &TokenStream2,
    vars: &[(String, Ident)],
    name: Ident,
    tracks: Vec<&String>,
    methods: Vec<TokenStream2>,
    runtime: TokenStream2,
    from_body: TokenStream2,
) -> TokenStream2 {
    let p = quote!(#krate::__private::prompt);
    let n = vars.len();
    let params: Vec<Ident> = (0..n).map(|i| format_ident!("__T{}", i)).collect();
    let fields: Vec<Ident> = (0..n).map(|i| format_ident!("__v{}", i)).collect();
    let traits: Vec<Ident> = vars
        .iter()
        .map(|(v, _)| format_ident!("Var_{}", v))
        .collect();
    let (generics, where_clause) = shape(vars);

    let all_names: Vec<&str> = vars.iter().map(|(v, _)| v.as_str()).collect();
    let listed = all_names
        .iter()
        .map(|v| format!("`{v}`"))
        .collect::<Vec<_>>()
        .join(", ");
    let trait_defs = vars.iter().zip(&traits).map(|((v, _), tr)| {
        let message = format!("prompt variable `{v}` is not set");
        let label = format!("call `.{v}(...)` before this");
        let note = format!("the template uses {listed}; each needs a value");
        quote! {
            #[diagnostic::on_unimplemented(message = #message, label = #label, note = #note)]
            #[allow(non_camel_case_types)]
            trait #tr: #p::Value {}
            impl #tr for #p::Set {}
        }
    });

    let setters = vars.iter().enumerate().map(|(i, (v, id))| {
        let out_params = params.iter().enumerate().map(|(j, t)| match j == i {
            true => quote!(#p::Set),
            false => quote!(#t),
        });
        let inits = fields.iter().enumerate().map(|(j, f)| {
            if j == i {
                quote!(#f: #p::set(value))
            } else {
                quote!(#f: self.#f)
            }
        });
        let doc = format!("Sets `{{{v}}}` to `value` (any `Display`).");
        quote! {
            #[doc = #doc]
            fn #id(self, value: impl ::core::fmt::Display) -> #name<#(#out_params),*> {
                #name { #(#inits),* }
            }
        }
    });

    let to_vars = {
        let inserts = vars.iter().zip(&fields).map(|((v, _), f)| {
            quote! {
                __m.insert(
                    ::std::string::String::from(#v),
                    ::std::string::String::from(#p::Value::get(&self.#f)),
                );
            }
        });
        quote! {
            /// The values as [`Vars`], to render a runtime template with them.
            fn to_vars(&self) -> #krate::Vars #where_clause {
                let mut __m = #krate::Vars::with_capacity(#n);
                #(#inserts)*
                __m
            }
        }
    };

    let struct_body = if n == 0 {
        quote!(;)
    } else {
        quote!({ #(#fields: #params),* })
    };
    let init = if n == 0 {
        quote!(#name)
    } else {
        quote!(#name { #(#fields: #p::Unset),* })
    };

    quote! {{
        #(const _: &str = ::core::include_str!(#tracks);)*
        #(#trait_defs)*

        /// Typed prompt generated by `prompt!` / `chat_prompt!`.
        #[derive(Clone, Debug)]
        #[allow(non_camel_case_types, dead_code)]
        struct #name #generics #struct_body

        #[allow(dead_code, non_camel_case_types)]
        impl #generics #name #generics {
            #(#setters)*

            /// Variable names in order of first appearance.
            fn variables(&self) -> &'static [&'static str] {
                &[#(#all_names),*]
            }

            #(#methods)*

            #to_vars
        }

        #[allow(non_camel_case_types)]
        impl #generics ::core::convert::From<#name #generics> for #runtime {
            #from_body
        }

        #init
    }}
}
