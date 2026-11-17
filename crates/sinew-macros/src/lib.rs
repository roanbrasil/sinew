//! Procedural macros for [Sinew](https://github.com/roanbrasil/sinew).
//!
//! Use them through `sinew` or `sinew-core` (feature `macros`), not directly:
//!
//! ```ignore
//! /// Looks up an order by id.
//! #[sinew::tool]
//! async fn get_order(
//!     /// Order id, e.g. "A-1001".
//!     id: String,
//! ) -> sinew::core::Result<Order> { /* ... */ }
//!
//! let tools = sinew::core::tools![get_order];
//! ```
//!
//! See the `tool` attribute in `sinew-core` for the full reference.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, spanned::Spanned, Attribute, Expr, ExprLit, FnArg, ItemFn, Lit, LitStr,
    Meta, Pat, Path, ReturnType, Type,
};

/// Turns a function into a Sinew `Tool`. See `sinew_core::tool` for docs.
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut args = ToolArgs::default();
    let parser = syn::meta::parser(|meta| {
        if meta.path.is_ident("name") {
            args.name = Some(meta.value()?.parse::<LitStr>()?);
            Ok(())
        } else if meta.path.is_ident("description") {
            args.description = Some(meta.value()?.parse::<LitStr>()?);
            Ok(())
        } else if meta.path.is_ident("crate") {
            let lit: LitStr = meta.value()?.parse()?;
            args.krate = Some(lit.parse::<Path>()?);
            Ok(())
        } else {
            Err(meta
                .error("unsupported #[tool] argument; expected `name`, `description` or `crate`"))
        }
    });
    parse_macro_input!(attr with parser);
    let func = parse_macro_input!(item as ItemFn);
    expand(args, func)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[derive(Default)]
struct ToolArgs {
    name: Option<LitStr>,
    description: Option<LitStr>,
    krate: Option<Path>,
}

/// Joins `#[doc = "..."]` lines the way rustdoc renders them.
fn doc_text(attrs: &[Attribute]) -> String {
    let lines: Vec<String> = attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .filter_map(|a| match &a.meta {
            Meta::NameValue(nv) => match &nv.value {
                Expr::Lit(ExprLit {
                    lit: Lit::Str(s), ..
                }) => Some(s.value()),
                _ => None,
            },
            _ => None,
        })
        .map(|l| l.strip_prefix(' ').unwrap_or(&l).trim_end().to_string())
        .collect();
    lines.join("\n").trim().to_string()
}

fn core_path() -> syn::Result<TokenStream2> {
    use proc_macro_crate::{crate_name, FoundCrate};
    match crate_name("sinew-core") {
        Ok(FoundCrate::Itself) => return Ok(quote!(::sinew_core)),
        Ok(FoundCrate::Name(n)) => {
            let id = format_ident!("{}", n);
            return Ok(quote!(::#id));
        }
        Err(_) => {}
    }
    match crate_name("sinew") {
        Ok(FoundCrate::Itself) => Ok(quote!(::sinew::core)),
        Ok(FoundCrate::Name(n)) => {
            let id = format_ident!("{}", n);
            Ok(quote!(::#id::core))
        }
        Err(_) => Err(syn::Error::new(
            Span::call_site(),
            "#[tool] needs `sinew` or `sinew-core` (feature `macros`) as a dependency; \
             or point to it with #[tool(crate = \"path::to::sinew_core\")]",
        )),
    }
}

/// True when the return type is spelled `Result<..>` (any path ending in `Result`).
fn returns_result(ret: &ReturnType) -> bool {
    match ret {
        ReturnType::Default => false,
        ReturnType::Type(_, ty) => match &**ty {
            Type::Path(p) => p.path.segments.last().is_some_and(|s| s.ident == "Result"),
            _ => false,
        },
    }
}

fn expand(args: ToolArgs, mut func: ItemFn) -> syn::Result<TokenStream2> {
    let sig = &func.sig;
    if !sig.generics.params.is_empty() || sig.generics.where_clause.is_some() {
        return Err(syn::Error::new(
            sig.generics.span(),
            "#[tool] functions cannot be generic: the model needs one concrete schema",
        ));
    }
    if let Some(v) = &sig.variadic {
        return Err(syn::Error::new(
            v.span(),
            "#[tool] functions cannot be variadic",
        ));
    }

    let krate = match args.krate {
        Some(p) => quote!(#p),
        None => core_path()?,
    };
    let fn_name = sig.ident.clone();
    let tool_name = args
        .name
        .map(|l| l.value())
        .unwrap_or_else(|| fn_name.to_string().trim_start_matches("r#").to_string());
    let description = match args.description {
        Some(d) => d.value(),
        None => doc_text(&func.attrs),
    };
    if description.is_empty() {
        return Err(syn::Error::new(
            fn_name.span(),
            "#[tool] needs a description: add a doc comment (`/// ...`) to the function \
             or use #[tool(description = \"...\")]",
        ));
    }

    let mut fields = Vec::new();
    let mut idents = Vec::new();
    // `#[state]` argument: injected by the application, hidden from the model.
    let mut state: Option<(syn::Ident, Type)> = None;
    let mut call_args = Vec::new();
    for input in func.sig.inputs.iter_mut() {
        let pt = match input {
            FnArg::Receiver(r) => {
                return Err(syn::Error::new(
                    r.span(),
                    "#[tool] works on free functions; pass shared state (a pool, a client) \
                     as an argument marked #[state] instead of `self`",
                ))
            }
            FnArg::Typed(pt) => pt,
        };
        let ident = match &*pt.pat {
            Pat::Ident(pi) if pi.subpat.is_none() && pi.by_ref.is_none() => pi.ident.clone(),
            other => {
                return Err(syn::Error::new(
                    other.span(),
                    "#[tool] arguments must be plain names like `id: String`",
                ))
            }
        };
        if let Type::Reference(r) = &*pt.ty {
            return Err(syn::Error::new(
                r.span(),
                "#[tool] arguments are deserialized from JSON and must be owned: \
                 use `String` instead of `&str`, `Vec<T>` instead of `&[T]`",
            ));
        }
        if let Some(pos) = pt.attrs.iter().position(|a| a.path().is_ident("state")) {
            let attr = pt.attrs.remove(pos);
            if state.is_some() {
                return Err(syn::Error::new(
                    attr.span(),
                    "#[tool] supports one #[state] argument; group shared state in a struct",
                ));
            }
            state = Some((ident.clone(), (*pt.ty).clone()));
            call_args.push(quote!(::core::clone::Clone::clone(&self.state)));
            continue;
        }
        // Doc comments and serde/schemars attributes describe the JSON field;
        // move them off the function parameter (where they are not allowed).
        let (field_attrs, keep): (Vec<Attribute>, Vec<Attribute>) =
            pt.attrs.drain(..).partition(|a| {
                a.path().is_ident("doc")
                    || a.path().is_ident("serde")
                    || a.path().is_ident("schemars")
            });
        pt.attrs = keep;
        let ty = &pt.ty;
        fields.push(quote! { #(#field_attrs)* #ident: #ty });
        call_args.push(quote!(#ident));
        idents.push(ident);
    }

    let vis = &func.vis;
    let call = if func.sig.asyncness.is_some() {
        quote!(#fn_name(#(#call_args),*).await)
    } else {
        quote!(#fn_name(#(#call_args),*))
    };
    let output = if returns_result(&func.sig.output) {
        quote! {
            match __out {
                ::core::result::Result::Ok(v) => #krate::__private::to_value(#tool_name, v),
                ::core::result::Result::Err(e) => ::core::result::Result::Err(#krate::__private::tool_error(#tool_name, e)),
            }
        }
    } else {
        quote!(#krate::__private::to_value(#tool_name, __out))
    };
    let serde_crate = format!("{}::__private::serde", krate.to_string().replace(' ', ""));
    let schemars_crate = format!(
        "{}::__private::schemars",
        krate.to_string().replace(' ', "")
    );
    let struct_doc = format!(
        "Tool generated by `#[tool]` from [`{fn_name}()`]: `{tool_name}` — {}",
        description.lines().next().unwrap_or_default()
    );

    let (tool_struct, constructor) = match &state {
        None => (
            quote! {
                #[doc = #struct_doc]
                #[allow(non_camel_case_types)]
                #[derive(Clone, Copy, Debug, Default)]
                #vis struct #fn_name {}
            },
            quote! {
                impl #fn_name {
                    /// The tool, ready for `ToolBox::with`.
                    #[allow(dead_code)]
                    pub fn tool() -> Self { Self {} }
                }
            },
        ),
        Some((_, ty)) => (
            quote! {
                #[doc = #struct_doc]
                #[allow(non_camel_case_types)]
                #[derive(Clone)]
                #vis struct #fn_name { state: #ty }
            },
            quote! {
                impl #fn_name {
                    /// The tool with its `#[state]`, ready for `ToolBox::with`.
                    #[allow(dead_code)]
                    pub fn tool(state: #ty) -> Self { Self { state } }
                }
            },
        ),
    };

    Ok(quote! {
        #func

        #tool_struct

        const _: () = {
            #[derive(#krate::__private::serde::Deserialize, #krate::__private::schemars::JsonSchema)]
            #[serde(crate = #serde_crate, deny_unknown_fields)]
            #[schemars(crate = #schemars_crate)]
            #[allow(non_camel_case_types)]
            struct __SinewToolArgs { #(#fields),* }

            #constructor

            #[#krate::__private::async_trait]
            impl #krate::Tool for #fn_name {
                fn spec(&self) -> #krate::ToolSpec {
                    #krate::ToolSpec {
                        name: ::std::string::String::from(#tool_name),
                        description: ::std::string::String::from(#description),
                        parameters: #krate::__private::schema_for::<__SinewToolArgs>(),
                    }
                }
                async fn call(&self, arguments: #krate::__private::serde_json::Value)
                    -> #krate::Result<#krate::__private::serde_json::Value>
                {
                    let __SinewToolArgs { #(#idents),* } =
                        #krate::__private::parse_args::<__SinewToolArgs>(#tool_name, arguments)?;
                    let __out = #call;
                    #output
                }
            }
        };
    })
}
