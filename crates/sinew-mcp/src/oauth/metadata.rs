//! Discovery documents and helpers shared by the OAuth client and server:
//! protected resource metadata (RFC 9728), authorization server metadata
//! (RFC 8414, with the OpenID Connect discovery fallback) and the
//! `WWW-Authenticate: Bearer` challenge (RFC 6750, RFC 9728 section 5.1).

use std::collections::HashMap;

use reqwest::Url;
use serde::{Deserialize, Serialize};

use crate::{error::Result, McpError};

/// Well-known path segment of protected resource metadata (RFC 9728).
pub const PROTECTED_RESOURCE_WELL_KNOWN: &str = "/.well-known/oauth-protected-resource";
/// Well-known path segment of authorization server metadata (RFC 8414).
pub const AUTHORIZATION_SERVER_WELL_KNOWN: &str = "/.well-known/oauth-authorization-server";
/// Well-known path segment of OpenID Connect discovery.
pub const OPENID_CONFIGURATION_WELL_KNOWN: &str = "/.well-known/openid-configuration";

/// OAuth 2.0 Protected Resource Metadata (RFC 9728), served by an MCP server
/// at `/.well-known/oauth-protected-resource`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtectedResourceMetadata {
    /// The resource identifier: the MCP server's canonical URL.
    pub resource: String,
    /// Issuer identifiers of the authorization servers that issue tokens for
    /// this resource.
    #[serde(default)]
    pub authorization_servers: Vec<String>,
    /// Scopes the resource understands.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scopes_supported: Vec<String>,
    /// How tokens may be presented (`header` only for MCP).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bearer_methods_supported: Vec<String>,
    /// Human-readable name of the resource.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource_name: Option<String>,
    /// URL of developer documentation for the resource.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource_documentation: Option<String>,
}

/// OAuth 2.0 Authorization Server Metadata (RFC 8414). Only the fields the
/// MCP flow uses are typed; the rest is kept in `extra`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthorizationServerMetadata {
    /// The issuer identifier; must equal the identifier the document was
    /// discovered from.
    pub issuer: String,
    /// Where the user is sent to authorize the client.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorization_endpoint: Option<String>,
    /// Where codes and refresh tokens are exchanged for tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_endpoint: Option<String>,
    /// The JSON Web Key Set that verifies the issuer's tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jwks_uri: Option<String>,
    /// Dynamic client registration endpoint (RFC 7591).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registration_endpoint: Option<String>,
    /// Scopes the server supports.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scopes_supported: Vec<String>,
    /// Supported `response_type` values.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub response_types_supported: Vec<String>,
    /// Supported grant types.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub grant_types_supported: Vec<String>,
    /// Supported PKCE methods (`S256` is required by this crate's client).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub code_challenge_methods_supported: Vec<String>,
    /// Supported client authentication methods at the token endpoint.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_endpoint_auth_methods_supported: Vec<String>,
    /// Every other metadata field.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// The parameters of a `WWW-Authenticate: Bearer ...` challenge.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BearerChallenge {
    /// Every auth-param, keyed by lowercase name.
    pub params: HashMap<String, String>,
}

impl BearerChallenge {
    /// Finds the `Bearer` challenge in a `WWW-Authenticate` header value
    /// (which may list several challenges). `None` when there is none.
    pub fn parse(header: &str) -> Option<Self> {
        parse_challenges(header)
            .into_iter()
            .find(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
            .map(|(_, params)| Self { params })
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.params.get(name).map(String::as_str)
    }

    /// `resource_metadata`: URL of the protected resource metadata (RFC 9728).
    pub fn resource_metadata(&self) -> Option<&str> {
        self.get("resource_metadata")
    }

    /// `scope`: scopes the request needs (space-separated).
    pub fn scope(&self) -> Option<&str> {
        self.get("scope")
    }

    /// `error`, e.g. `invalid_token` or `insufficient_scope`.
    pub fn error(&self) -> Option<&str> {
        self.get("error")
    }

    /// `error_description`.
    pub fn error_description(&self) -> Option<&str> {
        self.get("error_description")
    }
}

/// Parses `scheme param=value, param="quoted", other-scheme ...` into
/// `(scheme, params)` pairs. Token68 credentials are skipped.
fn parse_challenges(input: &str) -> Vec<(String, HashMap<String, String>)> {
    let mut out: Vec<(String, HashMap<String, String>)> = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    let skip_ws = |i: &mut usize| {
        while *i < chars.len() && (chars[*i] == ' ' || chars[*i] == '\t' || chars[*i] == ',') {
            *i += 1;
        }
    };
    let is_token_char = |c: char| !matches!(c, ' ' | '\t' | ',' | '=' | '"');
    loop {
        skip_ws(&mut i);
        if i >= chars.len() {
            break;
        }
        let start = i;
        while i < chars.len() && is_token_char(chars[i]) {
            i += 1;
        }
        let token: String = chars[start..i].iter().collect();
        if token.is_empty() {
            // Stray `=` or `"`: skip it.
            i += 1;
            continue;
        }
        let mut j = i;
        while j < chars.len() && (chars[j] == ' ' || chars[j] == '\t') {
            j += 1;
        }
        let is_param = j < chars.len() && chars[j] == '=' && chars.get(j + 1) != Some(&'=');
        if !is_param {
            // A new scheme (or a token68 credential, which then collects
            // no parameters and is ignored).
            out.push((token, HashMap::new()));
            continue;
        }
        i = j + 1;
        while i < chars.len() && (chars[i] == ' ' || chars[i] == '\t') {
            i += 1;
        }
        let mut value = String::new();
        if i < chars.len() && chars[i] == '"' {
            i += 1;
            while i < chars.len() && chars[i] != '"' {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    i += 1;
                }
                value.push(chars[i]);
                i += 1;
            }
            i += 1;
        } else {
            while i < chars.len() && chars[i] != ',' && chars[i] != ' ' && chars[i] != '\t' {
                value.push(chars[i]);
                i += 1;
            }
        }
        if let Some((_, params)) = out.last_mut() {
            params.entry(token.to_ascii_lowercase()).or_insert(value);
        }
    }
    out
}

/// Quotes a value for an auth-param (`"` and `\` escaped).
#[cfg(any(test, feature = "oauth-server"))]
pub(crate) fn quote(value: &str) -> String {
    let mut s = String::with_capacity(value.len() + 2);
    s.push('"');
    for c in value.chars() {
        if c == '"' || c == '\\' {
            s.push('\\');
        }
        s.push(c);
    }
    s.push('"');
    s
}

/// Parses an absolute `http(s)` URL without a fragment.
pub(crate) fn parse_url(what: &str, url: &str) -> Result<Url> {
    let parsed = Url::parse(url)
        .map_err(|e| McpError::Protocol(format!("invalid {what} URL `{url}`: {e}")))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(McpError::Protocol(format!(
            "invalid {what} URL `{url}`: expected http(s)"
        )));
    }
    if parsed.fragment().is_some() {
        return Err(McpError::Protocol(format!(
            "invalid {what} URL `{url}`: fragments are not allowed"
        )));
    }
    Ok(parsed)
}

/// True for `localhost`, `127.0.0.0/8` and `::1`.
pub(crate) fn is_loopback(url: &Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    bare.parse::<std::net::IpAddr>()
        .is_ok_and(|ip| ip.is_loopback())
}

/// OAuth 2.1 section 1.5: authorization server endpoints use HTTPS. Plain
/// HTTP is accepted for loopback hosts, or anywhere with `allow_http`.
pub(crate) fn check_secure(what: &str, url: &str, allow_http: bool) -> Result<Url> {
    let parsed = parse_url(what, url)?;
    if parsed.scheme() == "https" || allow_http || is_loopback(&parsed) {
        Ok(parsed)
    } else {
        Err(McpError::Protocol(format!(
            "{what} URL `{url}` must use https (plain http is only accepted for loopback hosts)"
        )))
    }
}

/// `{path}` of a URL without its trailing slash (`""` for the root).
fn trimmed_path(url: &Url) -> String {
    url.path().trim_end_matches('/').to_string()
}

/// `scheme://host[:port]` of a URL followed by `well_known` and `suffix`.
fn origin_join(url: &Url, well_known: &str, suffix: &str) -> String {
    let mut u = url.clone();
    u.set_path(&format!("{well_known}{suffix}"));
    u.set_query(None);
    u.set_fragment(None);
    u.to_string()
}

/// Where a resource's metadata lives (RFC 9728 section 3.1): the well-known
/// segment is inserted between the host and the resource's path, so
/// `https://api.example.com/mcp` maps to
/// `https://api.example.com/.well-known/oauth-protected-resource/mcp`.
pub fn protected_resource_metadata_url(resource: &str) -> Result<String> {
    let url = parse_url("resource", resource)?;
    Ok(origin_join(
        &url,
        PROTECTED_RESOURCE_WELL_KNOWN,
        &trimmed_path(&url),
    ))
}

/// Candidate metadata URLs for an issuer, in the order they are tried:
/// RFC 8414 (`/.well-known/oauth-authorization-server` inserted before the
/// issuer's path), then OpenID Connect discovery with the same insertion,
/// then OpenID Connect discovery appended to the issuer (Keycloak and most
/// OIDC providers).
pub fn authorization_server_metadata_urls(issuer: &str) -> Result<Vec<String>> {
    let url = parse_url("issuer", issuer)?;
    if url.query().is_some() {
        return Err(McpError::Protocol(format!(
            "invalid issuer `{issuer}`: query components are not allowed"
        )));
    }
    let path = trimmed_path(&url);
    let mut urls = vec![
        origin_join(&url, AUTHORIZATION_SERVER_WELL_KNOWN, &path),
        origin_join(&url, OPENID_CONFIGURATION_WELL_KNOWN, &path),
    ];
    if !path.is_empty() {
        urls.push(origin_join(
            &url,
            &format!("{path}{OPENID_CONFIGURATION_WELL_KNOWN}"),
            "",
        ));
    }
    Ok(urls)
}

/// Fetches and checks the metadata of `issuer`, trying
/// [`authorization_server_metadata_urls`] in order. The document's `issuer`
/// must equal `issuer` (RFC 8414 section 3.3).
pub async fn fetch_authorization_server_metadata(
    http: &reqwest::Client,
    issuer: &str,
    allow_http: bool,
) -> Result<AuthorizationServerMetadata> {
    check_secure("issuer", issuer, allow_http)?;
    let mut tried = Vec::new();
    for url in authorization_server_metadata_urls(issuer)? {
        let resp = match http
            .get(&url)
            .header(reqwest::header::ACCEPT, "application/json")
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tried.push(format!("{url}: {e}"));
                continue;
            }
        };
        let status = resp.status();
        if !status.is_success() {
            tried.push(format!("{url}: HTTP {status}"));
            continue;
        }
        let meta: AuthorizationServerMetadata = match resp.json().await {
            Ok(m) => m,
            Err(e) => {
                tried.push(format!("{url}: {e}"));
                continue;
            }
        };
        if meta.issuer != issuer {
            return Err(McpError::Protocol(format!(
                "authorization server metadata at {url} names issuer `{}`, expected `{issuer}`",
                meta.issuer
            )));
        }
        return Ok(meta);
    }
    Err(McpError::Protocol(format!(
        "no authorization server metadata for `{issuer}` ({})",
        tried.join("; ")
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bearer_challenges() {
        let c = BearerChallenge::parse(
            r#"Bearer error="insufficient_scope", scope="a b", resource_metadata="https://x/.well-known/oauth-protected-resource/mcp""#,
        )
        .unwrap();
        assert_eq!(c.error(), Some("insufficient_scope"));
        assert_eq!(c.scope(), Some("a b"));
        assert_eq!(
            c.resource_metadata(),
            Some("https://x/.well-known/oauth-protected-resource/mcp")
        );

        // Several challenges, unquoted values, escapes, other casing.
        let c = BearerChallenge::parse(
            r#"Basic realm="x", DPoP algs="ES256", bearer Error=invalid_token, error_description="say \"hi\"""#,
        )
        .unwrap();
        assert_eq!(c.error(), Some("invalid_token"));
        assert_eq!(c.error_description(), Some(r#"say "hi""#));
        assert!(BearerChallenge::parse(r#"Basic realm="x""#).is_none());
        assert_eq!(BearerChallenge::parse("Bearer").unwrap().params.len(), 0);
        assert!(BearerChallenge::parse("Negotiate abc==, Bearer realm=r")
            .unwrap()
            .params
            .contains_key("realm"));
    }

    #[test]
    fn quoting_round_trips() {
        let header = format!("Bearer error_description={}", quote(r#"a "b" \c"#));
        let c = BearerChallenge::parse(&header).unwrap();
        assert_eq!(c.error_description(), Some(r#"a "b" \c"#));
    }

    #[test]
    fn well_known_urls() {
        assert_eq!(
            protected_resource_metadata_url("https://api.example.com/mcp").unwrap(),
            "https://api.example.com/.well-known/oauth-protected-resource/mcp"
        );
        assert_eq!(
            protected_resource_metadata_url("https://api.example.com").unwrap(),
            "https://api.example.com/.well-known/oauth-protected-resource"
        );
        assert_eq!(
            authorization_server_metadata_urls("https://auth.example.com").unwrap(),
            [
                "https://auth.example.com/.well-known/oauth-authorization-server",
                "https://auth.example.com/.well-known/openid-configuration",
            ]
        );
        assert_eq!(
            authorization_server_metadata_urls("http://localhost:8080/realms/mcp").unwrap(),
            [
                "http://localhost:8080/.well-known/oauth-authorization-server/realms/mcp",
                "http://localhost:8080/.well-known/openid-configuration/realms/mcp",
                "http://localhost:8080/realms/mcp/.well-known/openid-configuration",
            ]
        );
        assert!(protected_resource_metadata_url("mcp.example.com").is_err());
        assert!(protected_resource_metadata_url("https://x/mcp#frag").is_err());
    }

    #[test]
    fn https_required_except_loopback() {
        assert!(check_secure("t", "https://auth.example.com/token", false).is_ok());
        assert!(check_secure("t", "http://localhost:8080/token", false).is_ok());
        assert!(check_secure("t", "http://127.0.0.1:8080/token", false).is_ok());
        assert!(check_secure("t", "http://[::1]:8080/token", false).is_ok());
        assert!(check_secure("t", "http://auth.example.com/token", false).is_err());
        assert!(check_secure("t", "http://auth.example.com/token", true).is_ok());
        assert!(check_secure("t", "ftp://auth.example.com/token", true).is_err());
    }
}
