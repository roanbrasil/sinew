//! Server-side bookkeeping for server-initiated notifications: one peer per
//! connection (stdio) or session (HTTP), each with an optional outbound
//! channel and its resource subscriptions.

use std::{
    collections::{HashMap, HashSet},
    sync::Mutex,
};

use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Default)]
struct Peer {
    /// `None` while no stream is attached (an HTTP session without GET).
    tx: Option<UnboundedSender<Value>>,
    subscriptions: HashSet<String>,
}

#[derive(Default)]
pub(crate) struct Hub {
    peers: Mutex<HashMap<String, Peer>>,
}

impl Hub {
    pub(crate) fn add_peer(&self, id: &str, tx: Option<UnboundedSender<Value>>) {
        self.peers.lock().unwrap().insert(
            id.to_string(),
            Peer {
                tx,
                subscriptions: HashSet::new(),
            },
        );
    }

    /// Attaches (or replaces) the outbound channel of a known peer.
    #[cfg_attr(not(feature = "http-server"), allow(dead_code))]
    pub(crate) fn attach(&self, id: &str, tx: UnboundedSender<Value>) -> bool {
        match self.peers.lock().unwrap().get_mut(id) {
            Some(p) => {
                p.tx = Some(tx);
                true
            }
            None => false,
        }
    }

    pub(crate) fn remove_peer(&self, id: &str) {
        self.peers.lock().unwrap().remove(id);
    }

    pub(crate) fn subscribe(&self, id: &str, uri: &str) {
        if let Some(p) = self.peers.lock().unwrap().get_mut(id) {
            p.subscriptions.insert(uri.to_string());
        }
    }

    pub(crate) fn unsubscribe(&self, id: &str, uri: &str) {
        if let Some(p) = self.peers.lock().unwrap().get_mut(id) {
            p.subscriptions.remove(uri);
        }
    }

    /// Sends `msg` to every attached peer that `filter` accepts and returns
    /// how many received it. A closed channel is detached.
    fn send_where(&self, msg: &Value, filter: impl Fn(&Peer) -> bool) -> usize {
        let mut sent = 0;
        for p in self.peers.lock().unwrap().values_mut() {
            if !filter(p) {
                continue;
            }
            if let Some(tx) = &p.tx {
                if tx.send(msg.clone()).is_ok() {
                    sent += 1;
                } else {
                    p.tx = None;
                }
            }
        }
        sent
    }

    pub(crate) fn send_to_subscribers(&self, uri: &str, msg: &Value) -> usize {
        self.send_where(msg, |p| p.subscriptions.contains(uri))
    }

    pub(crate) fn broadcast(&self, msg: &Value) -> usize {
        self.send_where(msg, |_| true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn routes_by_subscription_and_detaches_closed_streams() {
        let hub = Hub::default();
        let (tx_a, mut rx_a) = tokio::sync::mpsc::unbounded_channel();
        let (tx_b, rx_b) = tokio::sync::mpsc::unbounded_channel();
        hub.add_peer("a", Some(tx_a));
        hub.add_peer("b", None);
        assert!(hub.attach("b", tx_b));
        assert!(!hub.attach("zz", tokio::sync::mpsc::unbounded_channel().0));
        hub.subscribe("a", "x://1");
        assert_eq!(hub.send_to_subscribers("x://1", &json!(1)), 1);
        assert_eq!(rx_a.try_recv().unwrap(), json!(1));
        drop(rx_b);
        assert_eq!(hub.broadcast(&json!(2)), 1);
        hub.unsubscribe("a", "x://1");
        assert_eq!(hub.send_to_subscribers("x://1", &json!(3)), 0);
        hub.remove_peer("a");
        assert_eq!(hub.broadcast(&json!(4)), 0);
    }
}
