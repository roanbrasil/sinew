//! OAuth 2.1 authorization for the Streamable HTTP transport, following the
//! [MCP authorization specification](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization)
//! for protocol version `2025-06-18`.
//!
//! - Feature `oauth-server`: [`OAuthConfig`] for
//!   [`McpHttpConfig::with_oauth`](crate::McpHttpConfig::with_oauth). The
//!   endpoint serves protected resource metadata (RFC 9728), answers `401`
//!   with a `WWW-Authenticate` header pointing at it, validates JWT access
//!   tokens against the authorization server's JWKS (signature, issuer,
//!   audience, expiry) and enforces scopes per tool (`403 insufficient_scope`).
//!   [`JwtValidator`] is usable on its own.
//! - Feature `oauth-client`: [`OAuthClient`] discovers
//!   the authorization server from the MCP server's `401`, registers a client
//!   (RFC 7591) when needed, runs the authorization code flow with PKCE
//!   (RFC 7636) and the `resource` parameter (RFC 8707), refreshes tokens, and
//!   plugs into [`McpClient::connect_http_with_oauth`](crate::McpClient::connect_http_with_oauth)
//!   as a [`TokenProvider`].
//! - Feature `oauth`: both.
//!
//! [`metadata`] holds the discovery documents and helpers shared by both sides.
//!
//! Tokens are never logged or included in error messages, and the `Debug`
//! output of token-carrying types redacts them.

pub mod metadata;

#[cfg(feature = "oauth-client")]
pub mod client;
#[cfg(feature = "oauth-server")]
pub mod server;

pub use metadata::{AuthorizationServerMetadata, BearerChallenge, ProtectedResourceMetadata};

#[cfg(feature = "oauth-client")]
pub use client::{
    ClientRegistration, InMemoryTokenStore, OAuthClient, OAuthClientConfig, Pkce, TokenProvider,
    TokenSet, TokenStore,
};
#[cfg(feature = "oauth-server")]
pub use server::{AccessClaims, JwtValidator, OAuthConfig, TokenError};

/// Byte-wise comparison whose running time does not depend on where the
/// inputs differ (only on their lengths).
#[cfg(feature = "oauth-client")]
pub(crate) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}
