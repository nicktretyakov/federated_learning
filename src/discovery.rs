use anyhow::Result;
use etcd_client::{Client, PutOptions};
use log::{error, info};
use std::time::Duration;
use tokio::time;

// Register a node with etcd service discovery
pub async fn register_node(
    etcd_endpoints: &[String],
    node_id: &str,
    node_addr: &str,
) -> Result<()> {
    let mut client = Client::connect(etcd_endpoints, None).await?;

    let key = format!("/fedlearn/nodes/{}", node_id);
    let lease_id = client.lease_grant(30, None).await?.id();

    let put_options = PutOptions::new().with_lease(lease_id);
    client
        .put(key, node_addr.to_string(), Some(put_options))
        .await?;

    info!("Registered node {} at {} in etcd", node_id, node_addr);

    // Keep lease alive
    let mut client_clone = client.clone();
    let lease_id_clone = lease_id;

    tokio::spawn(async move {
        let mut interval = time::interval(Duration::from_secs(10));
        loop {
            interval.tick().await;
            match client_clone.lease_keep_alive(lease_id_clone).await {
                Ok(_) => info!("Node registration lease renewed"),
                Err(e) => {
                    error!("Failed to renew node registration lease: {}", e);
                    break;
                }
            }
        }
    });

    Ok(())
}

// Discover all nodes from etcd
pub async fn discover_nodes(etcd_endpoints: &[String]) -> Result<Vec<String>> {
    let mut client = Client::connect(etcd_endpoints, None).await?;

    let response = client
        .get(
            "/fedlearn/nodes/",
            Some(etcd_client::GetOptions::new().with_prefix()),
        )
        .await?;

    let mut nodes = Vec::new();
    for kv in response.kvs() {
        let value = kv.value_str()?;
        nodes.push(value.to_string());
    }

    info!("Discovered {} nodes", nodes.len());
    Ok(nodes)
}
