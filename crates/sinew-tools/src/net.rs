//! Guard for tools that open URLs chosen by the model (server-side request
//! forgery protection).
//!
//! A [`HostPolicy`] decides which URLs a tool may open:
//!
//! - only `http` and `https`;
//! - a deny-list and an optional allow-list of host names (a name also
//!   matches its subdomains);
//! - by default no loopback, private, link-local, shared (CGNAT), multicast,
//!   documentation or otherwise reserved address, whether written as an IP
//!   literal or reached through DNS.
//!
//! [`HostPolicy::client_builder`] returns a `reqwest` client builder that
//! enforces the policy at connection time: host names are resolved by a
//! resolver that drops non-public addresses (so a name that resolves to
//! `10.0.0.5` or `169.254.169.254` is refused, and DNS rebinding between the
//! check and the connection does not help), every redirect target is checked
//! again, and proxies from the environment are ignored (a proxy would resolve
//! the name itself).
//!
//! ```
//! use sinew_tools::net::HostPolicy;
//!
//! let policy = HostPolicy::new().deny_host("example.org");
//! assert!(policy.check_url_str("https://docs.rs/serde").is_ok());
//! assert!(policy.check_url_str("http://127.0.0.1:8080/admin").is_err());
//! assert!(policy.check_url_str("http://169.254.169.254/latest/meta-data").is_err());
//! assert!(policy.check_url_str("https://www.example.org/").is_err());
//! assert!(policy.check_url_str("file:///etc/passwd").is_err());
//! ```

use std::{
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    sync::Arc,
};

use reqwest::dns::{Addrs, Name, Resolve, Resolving};

use crate::ToolError;

/// Prefix of resolver and redirect errors, so [`ToolError::from_reqwest`] can
/// report them as [`ToolError::Blocked`].
pub(crate) const BLOCKED_MARKER: &str = "sinew-blocked: ";

/// Redirects followed at most by a guarded client.
pub const MAX_REDIRECTS: usize = 5;

/// Which hosts and addresses a tool may connect to. See the
/// [module docs](self).
///
/// ```
/// use sinew_tools::net::HostPolicy;
///
/// let policy = HostPolicy::new().allow_hosts(["api.example.com", "docs.rs"]);
/// assert!(policy.check_url_str("https://eu.api.example.com/v1").is_ok());
/// assert!(policy.check_url_str("https://example.com/").is_err());
/// let client = policy.client_builder().build().unwrap();
/// # let _ = client;
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HostPolicy {
    allow_hosts: Vec<String>,
    deny_hosts: Vec<String>,
    allow_private: bool,
    pinned: Vec<(String, Vec<IpAddr>)>,
}

fn normalize_host(host: &str) -> String {
    host.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim_end_matches('.')
        .to_ascii_lowercase()
}

/// `host` equals `pattern` or is a subdomain of it.
fn host_matches(host: &str, pattern: &str) -> bool {
    host == pattern
        || (host.len() > pattern.len()
            && host.ends_with(pattern)
            && host.as_bytes()[host.len() - pattern.len() - 1] == b'.')
}

impl HostPolicy {
    /// Any public host; no private addresses.
    pub fn new() -> Self {
        Self::default()
    }

    /// Restricts the policy to this host and its subdomains. Once one host is
    /// allowed, every host not on the list is refused. Allowing a host does
    /// not allow private addresses; see [`allow_private`](Self::allow_private).
    pub fn allow_host(mut self, host: impl AsRef<str>) -> Self {
        self.allow_hosts.push(normalize_host(host.as_ref()));
        self
    }

    /// [`allow_host`](Self::allow_host) for several hosts.
    pub fn allow_hosts<I, S>(mut self, hosts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for h in hosts {
            self = self.allow_host(h);
        }
        self
    }

    /// Refuses this host and its subdomains, even when it is allowed.
    pub fn deny_host(mut self, host: impl AsRef<str>) -> Self {
        self.deny_hosts.push(normalize_host(host.as_ref()));
        self
    }

    /// Allows loopback, private and link-local addresses (default `false`).
    /// Turn it on only for tools that must reach an intranet or a local
    /// service, ideally together with an allow-list.
    pub fn allow_private(mut self, on: bool) -> Self {
        self.allow_private = on;
        self
    }

    /// Resolves `host` to fixed addresses instead of asking DNS (split-horizon
    /// setups, tests). The addresses still pass the private-address check.
    pub fn pin_host(
        mut self,
        host: impl AsRef<str>,
        addrs: impl IntoIterator<Item = IpAddr>,
    ) -> Self {
        self.pinned
            .push((normalize_host(host.as_ref()), addrs.into_iter().collect()));
        self
    }

    /// The allow-list (empty: any public host).
    pub fn allowed_hosts(&self) -> &[String] {
        &self.allow_hosts
    }

    /// The deny-list.
    pub fn denied_hosts(&self) -> &[String] {
        &self.deny_hosts
    }

    /// Whether private addresses are allowed.
    pub fn allows_private(&self) -> bool {
        self.allow_private
    }

    /// Checks an address (used for IP literals and for every DNS answer).
    pub fn check_ip(&self, ip: IpAddr) -> Result<(), ToolError> {
        if self.allow_private || is_public_ip(ip) {
            Ok(())
        } else {
            Err(ToolError::Blocked(format!(
                "address {ip} is loopback, private, link-local or reserved"
            )))
        }
    }

    /// Checks scheme, host lists and IP literals. Host names are resolved
    /// later, by the client from [`client_builder`](Self::client_builder).
    pub fn check_url(&self, url: &reqwest::Url) -> Result<(), ToolError> {
        if !matches!(url.scheme(), "http" | "https") {
            return Err(ToolError::Blocked(format!(
                "scheme `{}` is not allowed; use http or https",
                url.scheme()
            )));
        }
        let host = match url.host() {
            Some(url::Host::Domain(d)) => normalize_host(d),
            Some(url::Host::Ipv4(ip)) => ip.to_string(),
            Some(url::Host::Ipv6(ip)) => ip.to_string(),
            None => return Err(ToolError::Invalid("the URL has no host".into())),
        };
        if self.deny_hosts.iter().any(|p| host_matches(&host, p)) {
            return Err(ToolError::Blocked(format!(
                "host `{host}` is on the deny-list"
            )));
        }
        if !self.allow_hosts.is_empty() && !self.allow_hosts.iter().any(|p| host_matches(&host, p))
        {
            return Err(ToolError::Blocked(format!(
                "host `{host}` is not on the allow-list ({})",
                self.allow_hosts.join(", ")
            )));
        }
        match url.host() {
            Some(url::Host::Ipv4(ip)) => self.check_ip(IpAddr::V4(ip))?,
            Some(url::Host::Ipv6(ip)) => self.check_ip(IpAddr::V6(ip))?,
            _ => {
                if !self.allow_private
                    && (host == "localhost"
                        || host.ends_with(".localhost")
                        || host.ends_with(".local")
                        || host.ends_with(".internal"))
                {
                    return Err(ToolError::Blocked(format!("host `{host}` is a local name")));
                }
            }
        }
        Ok(())
    }

    /// Parses `url` and runs [`check_url`](Self::check_url).
    pub fn check_url_str(&self, url: &str) -> Result<reqwest::Url, ToolError> {
        let parsed = reqwest::Url::parse(url)
            .map_err(|e| ToolError::Invalid(format!("bad URL `{url}`: {e}")))?;
        self.check_url(&parsed)?;
        Ok(parsed)
    }

    /// A `reqwest` client builder that enforces this policy on every
    /// connection: guarded DNS resolution, redirects re-checked (at most
    /// [`MAX_REDIRECTS`]), environment proxies ignored. Add timeouts and a
    /// user agent, then `build()`.
    pub fn client_builder(&self) -> reqwest::ClientBuilder {
        let policy = Arc::new(self.clone());
        let redirect_policy = policy.clone();
        reqwest::Client::builder()
            .dns_resolver(Arc::new(GuardedResolver { policy }))
            .no_proxy()
            .redirect(reqwest::redirect::Policy::custom(move |attempt| {
                if attempt.previous().len() >= MAX_REDIRECTS {
                    return attempt.error(format!("more than {MAX_REDIRECTS} redirects"));
                }
                match redirect_policy.check_url(attempt.url()) {
                    Ok(()) => attempt.follow(),
                    Err(ToolError::Blocked(m)) => {
                        attempt.error(format!("{BLOCKED_MARKER}redirect refused: {m}"))
                    }
                    Err(e) => attempt.error(e.to_string()),
                }
            }))
    }

    fn pinned(&self, host: &str) -> Option<&[IpAddr]> {
        self.pinned
            .iter()
            .find(|(h, _)| h == host)
            .map(|(_, a)| a.as_slice())
    }
}

/// DNS resolver that drops addresses the policy refuses.
struct GuardedResolver {
    policy: Arc<HostPolicy>,
}

impl Resolve for GuardedResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let policy = self.policy.clone();
        let host = normalize_host(name.as_str());
        Box::pin(async move {
            let addrs: Vec<SocketAddr> = match policy.pinned(&host) {
                Some(ips) => ips.iter().map(|ip| SocketAddr::new(*ip, 0)).collect(),
                None => tokio::net::lookup_host((host.as_str(), 0)).await?.collect(),
            };
            let allowed: Vec<SocketAddr> = addrs
                .into_iter()
                .filter(|a| policy.check_ip(a.ip()).is_ok())
                .collect();
            if allowed.is_empty() {
                return Err(format!(
                    "{BLOCKED_MARKER}host `{host}` resolves only to loopback, private, link-local or reserved addresses"
                )
                .into());
            }
            Ok(Box::new(allowed.into_iter()) as Addrs)
        })
    }
}

/// True for a globally routable unicast address. False for loopback,
/// private (RFC 1918, IPv6 unique local), link-local (including the cloud
/// metadata address `169.254.169.254`), unspecified, shared (CGNAT,
/// `100.64.0.0/10`), benchmarking, documentation, multicast, broadcast and
/// reserved ranges. IPv4-mapped, IPv4-compatible, NAT64 and 6to4 IPv6
/// addresses are judged by the IPv4 address they embed; Teredo is refused.
///
/// ```
/// use sinew_tools::net::is_public_ip;
/// assert!(is_public_ip("93.184.215.14".parse().unwrap()));
/// assert!(!is_public_ip("10.1.2.3".parse().unwrap()));
/// assert!(!is_public_ip("::ffff:127.0.0.1".parse().unwrap()));
/// ```
pub fn is_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_public_v4(v4),
        IpAddr::V6(v6) => is_public_v6(v6),
    }
}

fn is_public_v4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_unspecified()
        || ip.is_broadcast()
        || ip.is_multicast()
        || ip.is_documentation()
        || a == 0
        || (a == 100 && (b & 0xc0) == 64) // 100.64.0.0/10 shared address space
        || (a == 192 && b == 0 && c == 0) // 192.0.0.0/24 IETF protocol assignments
        || (a == 198 && (b & 0xfe) == 18) // 198.18.0.0/15 benchmarking
        || a >= 240) // 240.0.0.0/4 reserved
}

fn is_public_v6(ip: Ipv6Addr) -> bool {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return is_public_v4(v4);
    }
    let s = ip.segments();
    let embedded =
        |hi: u16, lo: u16| Ipv4Addr::new((hi >> 8) as u8, hi as u8, (lo >> 8) as u8, lo as u8);
    // ::a.b.c.d (deprecated IPv4-compatible); `::` and `::1` are covered below.
    if s[..6] == [0; 6] && !(s[6] == 0 && s[7] <= 1) {
        return is_public_v4(embedded(s[6], s[7]));
    }
    // 64:ff9b::/96 NAT64.
    if s[..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
        return is_public_v4(embedded(s[6], s[7]));
    }
    // 2002::/16 6to4.
    if s[0] == 0x2002 {
        return is_public_v4(embedded(s[1], s[2]));
    }
    !(ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_multicast()
        || (s[0] & 0xfe00) == 0xfc00 // fc00::/7 unique local
        || (s[0] & 0xffc0) == 0xfe80 // fe80::/10 link-local
        || (s[0] & 0xffc0) == 0xfec0 // fec0::/10 site-local (deprecated)
        || (s[0] == 0x2001 && s[1] == 0xdb8) // documentation
        || (s[0] == 0x2001 && s[1] == 0) // Teredo
        || (s[0] == 0x64 && s[1] == 0xff9b && s[2] == 1) // 64:ff9b:1::/48 local NAT64
        || (s[0] == 0x100 && s[1..4] == [0, 0, 0])) // 100::/64 discard
}

/// Reads at most `max_bytes` of the body; the flag is true when more was
/// available.
#[allow(dead_code)]
pub(crate) async fn read_limited(
    mut resp: reqwest::Response,
    max_bytes: usize,
) -> Result<(Vec<u8>, bool), ToolError> {
    let mut body = Vec::new();
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| ToolError::from_reqwest(&e))?
    {
        let room = max_bytes.saturating_sub(body.len());
        if chunk.len() > room {
            body.extend_from_slice(&chunk[..room]);
            return Ok((body, true));
        }
        body.extend_from_slice(&chunk);
    }
    Ok((body, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocked(p: &HostPolicy, url: &str) -> bool {
        matches!(p.check_url_str(url), Err(ToolError::Blocked(_)))
    }

    #[test]
    fn classifies_addresses() {
        for ip in [
            "127.0.0.1",
            "127.255.0.9",
            "10.0.0.1",
            "172.16.5.4",
            "192.168.1.1",
            "169.254.169.254",
            "0.0.0.0",
            "0.1.2.3",
            "100.64.0.1",
            "198.18.0.1",
            "192.0.2.10",
            "224.0.0.1",
            "255.255.255.255",
            "240.0.0.1",
            "::1",
            "::",
            "fc00::1",
            "fd12:3456::1",
            "fe80::1",
            "ff02::1",
            "2001:db8::1",
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.1",
            "::ffff:169.254.169.254",
            "::127.0.0.1",
            "64:ff9b::a00:1",
            "2002:a00:1::",
            "2001:0:4136:e378::1",
        ] {
            assert!(!is_public_ip(ip.parse().unwrap()), "{ip} should be blocked");
        }
        for ip in [
            "93.184.215.14",
            "8.8.8.8",
            "1.1.1.1",
            "2606:4700:4700::1111",
            "::ffff:8.8.8.8",
            "64:ff9b::808:808",
        ] {
            assert!(is_public_ip(ip.parse().unwrap()), "{ip} should be public");
        }
    }

    #[test]
    fn checks_urls() {
        let p = HostPolicy::new();
        for url in [
            "http://127.0.0.1/",
            "http://127.0.0.1:8080/x",
            "http://10.1.1.1/",
            "http://[::1]/",
            "http://[::ffff:7f00:1]/",
            "http://169.254.169.254/latest/meta-data/",
            "http://localhost:3000/",
            "http://LOCALHOST./",
            "http://api.localhost/",
            "http://printer.local/",
            "http://metadata.google.internal/",
            "http://2130706433/",
            "http://0x7f.1/",
            "file:///etc/passwd",
            "ftp://example.com/",
            "gopher://example.com/",
        ] {
            assert!(blocked(&p, url), "{url} should be blocked");
        }
        assert!(p.check_url_str("https://example.com/a?b=c").is_ok());
        assert!(matches!(
            p.check_url_str("not a url"),
            Err(ToolError::Invalid(_))
        ));

        let p = HostPolicy::new()
            .allow_host("Docs.RS")
            .deny_host("bad.docs.rs");
        assert!(p.check_url_str("https://docs.rs/x").is_ok());
        assert!(p.check_url_str("https://sub.docs.rs/x").is_ok());
        assert!(blocked(&p, "https://bad.docs.rs/"));
        assert!(blocked(&p, "https://x.bad.docs.rs/"));
        assert!(blocked(&p, "https://evildocs.rs/"));
        assert!(blocked(&p, "https://docs.rs.evil.com/"));
        assert!(blocked(&p, "https://evil.com/docs.rs"));

        let p = HostPolicy::new().allow_private(true);
        assert!(p.check_url_str("http://127.0.0.1:8080/").is_ok());
        assert!(p.check_url_str("http://localhost/").is_ok());
    }

    async fn get(policy: &HostPolicy, url: &str) -> Result<reqwest::Response, ToolError> {
        let client = policy.client_builder().build().unwrap();
        let url = policy.check_url_str(url)?;
        client
            .get(url)
            .send()
            .await
            .map_err(|e| ToolError::from_reqwest(&e))
    }

    #[tokio::test]
    async fn resolver_blocks_names_that_resolve_to_private_addresses() {
        for ip in ["127.0.0.1", "10.0.0.5", "169.254.169.254", "::1", "fd00::1"] {
            let p = HostPolicy::new().pin_host("internal.example.com", [ip.parse().unwrap()]);
            let err = get(&p, "http://internal.example.com/").await.unwrap_err();
            assert!(
                matches!(&err, ToolError::Blocked(m) if m.contains("internal.example.com")),
                "{ip}: {err}"
            );
        }
    }

    #[tokio::test]
    async fn resolver_allows_names_when_private_is_allowed() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            while let Ok((mut s, _)) = listener.accept().await {
                let mut buf = [0u8; 1024];
                let _ = s.read(&mut buf).await;
                let _ = s
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok",
                    )
                    .await;
            }
        });
        let pinned = |p: HostPolicy| p.pin_host("svc.example.com", ["127.0.0.1".parse().unwrap()]);
        let url = format!("http://svc.example.com:{port}/");
        let err = get(&pinned(HostPolicy::new()), &url).await.unwrap_err();
        assert!(matches!(err, ToolError::Blocked(_)), "{err}");
        let resp = get(&pinned(HostPolicy::new().allow_private(true)), &url)
            .await
            .unwrap();
        assert_eq!(resp.text().await.unwrap(), "ok");
    }

    #[tokio::test]
    async fn redirects_to_private_addresses_are_refused() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            while let Ok((mut s, _)) = listener.accept().await {
                let mut buf = [0u8; 1024];
                let _ = s.read(&mut buf).await;
                let _ = s
                    .write_all(b"HTTP/1.1 302 Found\r\nLocation: http://169.254.169.254/latest/\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    .await;
            }
        });
        // The test server listens on loopback, so private addresses are
        // allowed here; the redirect target is refused because every hop is
        // checked against the host lists again.
        let url = format!("http://127.0.0.1:{port}/");
        for p in [
            HostPolicy::new()
                .allow_private(true)
                .deny_host("169.254.169.254"),
            HostPolicy::new()
                .allow_private(true)
                .allow_host("127.0.0.1"),
        ] {
            let err = get(&p, &url).await.unwrap_err();
            assert!(
                matches!(&err, ToolError::Blocked(m) if m.contains("redirect")),
                "{err}"
            );
        }
    }
}
