mod discovery;
mod messages;
mod model;
mod network;
mod node;
mod server;

use actix::Actor;
use actix_web::{middleware, web, App, HttpServer};
use anyhow::Result;
use env_logger::Env;
use log::{error, info};
use node::NodeActor;
use once_cell::sync::Lazy;
use server::CentralServer;
use std::env;

// Global server address for access throughout the app
static SERVER_ADDR: Lazy<String> = Lazy::new(|| {
    env::var("SERVER_ADDR").unwrap_or_else(|_| "http://127.0.0.1:5000".to_string())
});

// Number of nodes to expect
static TOTAL_NODES: Lazy<usize> = Lazy::new(|| {
    env::var("TOTAL_NODES")
        .unwrap_or_else(|_| "2".to_string())
        .parse()
        .unwrap_or(2)
});

#[actix_web::main]
async fn main() -> Result<()> {
    // Initialize logger
    env_logger::init_from_env(Env::default().default_filter_or("info"));

    // Determine if we're running as a server or node
    let is_server = env::var("RUN_AS").unwrap_or_else(|_| "server".to_string()) == "server";

    if is_server {
        run_server().await?;
    } else {
        let node_id = env::var("NODE_ID").unwrap_or_else(|_| "node1".to_string());
        let node_addr = format!("http://127.0.0.1:{}", 8001 + node_id.parse::<u16>().unwrap_or(1));
        run_node(&node_id, &node_addr).await?;
    }

    Ok(())
}

async fn run_server() -> Result<()> {
    info!("Starting central server");

    // Start central server actor
    let server_actor = CentralServer::new(*TOTAL_NODES).start();

    // Start HTTP server for API endpoints
    let server = HttpServer::new(move || {
        App::new()
            .wrap(middleware::Logger::default())
            .app_data(web::Data::new(server_actor.clone()))
            .route("/message", web::post().to(network::receive_server_message))
            .route("/status", web::get().to(network::get_server_status))
            .route("/", web::get().to(|| async {
                actix_web::HttpResponse::Ok()
                    .content_type("text/html")
                    .body(include_str!("../templates/dashboard.html"))
            }))
            .route("/api/nodes", web::get().to(network::get_all_nodes))
            .route("/api/model/params", web::get().to(network::get_model_params))
    })
    .bind(("0.0.0.0", 5000))?
    .run();

    info!("Central server listening on 0.0.0.0:5000");

    server.await?;
    Ok(())
}

async fn run_node(node_id: &str, node_addr: &str) -> Result<()> {
    info!("Starting node {} at {}", node_id, node_addr);

    // Parse port from node_addr
    let addr_parts: Vec<&str> = node_addr.split(':').collect();
    let port = if addr_parts.len() > 2 {
        addr_parts[2].parse::<u16>().unwrap_or(8001)
    } else {
        8001
    };

    // Start node actor
    let node_actor = NodeActor::new(SERVER_ADDR.clone(), node_addr.to_string()).start();

    // Optional: Register with etcd if ETCD_ENDPOINTS is set
    if let Ok(etcd_endpoints) = env::var("ETCD_ENDPOINTS") {
        let endpoints: Vec<String> = etcd_endpoints.split(',').map(String::from).collect();
        if !endpoints.is_empty() {
            match discovery::register_node(&endpoints, node_id, node_addr).await {
                Ok(_) => info!("Registered node with etcd"),
                Err(e) => error!("Failed to register node with etcd: {}", e),
            }
        }
    }

    // Generate some train data and train the model
    let (data, labels) = model::generate_data(100);
    let _ = node_actor.send(messages::NodeMessage::Train {
        data: data.clone(),
        labels: labels.clone()
    }).await;

    // Start HTTP server for API endpoints
    let server = HttpServer::new(move || {
        App::new()
            .wrap(middleware::Logger::default())
            .app_data(web::Data::new(node_actor.clone()))
            .route("/message", web::post().to(network::receive_node_message))
            .route("/status", web::get().to(network::get_node_status))
            .route("/train", web::post().to(|data: web::Json<(Vec<f32>, Vec<f32>)>, actor: web::Data<actix::Addr<NodeActor>>| {
                async move {
                    let (data, labels) = data.into_inner();
                    match actor.send(messages::NodeMessage::Train { data, labels }).await {
                        Ok(Ok(())) => "Training started",
                        _ => "Failed to start training",
                    }
                }
            }))
    })
    .bind(("0.0.0.0", port))?
    .run();

    info!("Node listening on 0.0.0.0:{}", port);

    server.await?;
    Ok(())
}
