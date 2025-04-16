use crate::messages::{GetModelParams, GetNodesRequest, NodeMessage};
use crate::node::NodeActor;
use crate::server::CentralServer;
use actix::Addr;
use actix_web::{web, HttpResponse, Responder};
use log::{error, info};
use serde::{Deserialize, Serialize};

// Handler for receiving messages at nodes
pub async fn receive_node_message(
    msg: web::Json<NodeMessage>,
    actor: web::Data<Addr<NodeActor>>,
) -> impl Responder {
    info!("Node received message: {:?}", msg.0);

    match actor.send(msg.0).await {
        Ok(Ok(())) => HttpResponse::Ok().json(serde_json::json!({"status": "success"})),
        Ok(Err(e)) => {
            error!("Error handling message: {}", e);
            HttpResponse::InternalServerError()
                .json(serde_json::json!({"status": "error", "message": e}))
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            HttpResponse::InternalServerError()
                .json(serde_json::json!({"status": "error", "message": e.to_string()}))
        }
    }
}

// Handler for receiving messages at server
pub async fn receive_server_message(
    msg: web::Json<NodeMessage>,
    server: web::Data<Addr<CentralServer>>,
) -> impl Responder {
    info!("Server received message: {:?}", msg.0);

    match server.send(msg.0).await {
        Ok(Ok(())) => HttpResponse::Ok().json(serde_json::json!({"status": "success"})),
        Ok(Err(e)) => {
            error!("Error handling message at server: {}", e);
            HttpResponse::InternalServerError()
                .json(serde_json::json!({"status": "error", "message": e}))
        }
        Err(e) => {
            error!("Server actor mailbox error: {}", e);
            HttpResponse::InternalServerError()
                .json(serde_json::json!({"status": "error", "message": e.to_string()}))
        }
    }
}

// API response for node status
#[derive(Serialize, Deserialize, Clone)]
pub struct NodeStatus {
    pub address: String,
    pub status: String,
}

// Handler for getting node status
pub async fn get_node_status(actor: web::Data<Addr<NodeActor>>) -> impl Responder {
    match actor
        .send(NodeMessage::Predict {
            data: vec![0.0; 10],
        })
        .await
    {
        Ok(_) => {
            let node_status = NodeStatus {
                address: "active".to_string(),
                status: "running".to_string(),
            };
            HttpResponse::Ok().json(node_status)
        }
        Err(e) => {
            error!("Node status error: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "status": "error",
                "message": format!("Failed to get node status: {}", e)
            }))
        }
    }
}

// Handler for getting server status
pub async fn get_server_status(server: web::Data<Addr<CentralServer>>) -> impl Responder {
    // Simple ping to check if server is responsive
    match server
        .send(NodeMessage::RegisterNode {
            addr: "ping".to_string(),
        })
        .await
    {
        Ok(_) => HttpResponse::Ok().json(serde_json::json!({
            "status": "running",
            "message": "Server is active"
        })),
        Err(e) => {
            error!("Server status error: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "status": "error",
                "message": format!("Failed to get server status: {}", e)
            }))
        }
    }
}

// Handler for getting all connected nodes
pub async fn get_all_nodes(server: web::Data<Addr<CentralServer>>) -> impl Responder {
    match server.send(GetNodesRequest).await {
        Ok(nodes) => {
            info!("Returning information about {} nodes", nodes.len());
            HttpResponse::Ok().json(nodes)
        }
        Err(e) => {
            error!("Failed to get nodes information: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "status": "error",
                "message": format!("Failed to get nodes information: {}", e)
            }))
        }
    }
}

// Handler for getting model parameters
pub async fn get_model_params(server: web::Data<Addr<CentralServer>>) -> impl Responder {
    match server.send(GetModelParams).await {
        Ok(Ok(params)) => {
            info!("Returning model parameters, size: {}", params.len());
            HttpResponse::Ok().json(serde_json::json!({
                "status": "success",
                "parameters": params,
                "size": params.len()
            }))
        }
        Ok(Err(e)) => {
            error!("Failed to get model parameters: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "status": "error",
                "message": e
            }))
        }
        Err(e) => {
            error!("Failed to communicate with server actor: {}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "status": "error",
                "message": format!("Failed to communicate with server: {}", e)
            }))
        }
    }
}
