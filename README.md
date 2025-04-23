# Federated Learning in Rust

A lightweight federated learning framework implemented in Rust using **Actix** and **ndarray**, demonstrating a simple FedAvg workflow with a central server and multiple training nodes.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Usage](#usage)
   - [Running the Central Server](#running-the-central-server)
   - [Running a Node](#running-a-node)
   - [With etcd Service Discovery (Optional)](#with-etcd-service-discovery-optional)
8. [API Reference](#api-reference)
9. [Code Structure](#code-structure)
10. [Model Details](#model-details)
11. [Dashboard](#dashboard)
12. [Logging & Monitoring](#logging--monitoring)
13. [Contributing](#contributing)

---

## Overview

This project implements a federated learning setup in Rust, comprised of:

- **Central Server**: Aggregates model updates from nodes and redistributes the global model.
- **Node Actors**: Train models locally on synthetic data and send parameter updates to the server.
- **FedAvg Algorithm**: Simple averaging of model parameters over all participating nodes.
- **Asynchronous Communication**: Built on the Actix actor framework and actix-web for HTTP APIs.

## Features

- 🔄 **FedAvg aggregation** of model parameters.
- 🧩 **Modular codebase**: clear separation of discovery, messaging, model, network, node, and server.
- 🌐 **HTTP API endpoints** for messaging, status checks, and model inspection.
- 📊 **Dashboard UI** to monitor server status, connected nodes, and model parameters.
- 🔧 **Optional etcd-based service discovery** for dynamic node registration.

## Architecture

```text
+-------------------+                      +-------------------+
|  Node 1 (Rust)   | --POST /message-->   | Central Server    |
|  Node Actor       |                      | Actor             |
+-------------------+ <---updates-------- +-------------------+
         ^                                         |
         |                                         v
         +--Local training & FedAvg updates------->

+-------------------+
|  Node 2 (Rust)    |
+-------------------+

... up to TOTAL_NODES ====================================
```

1. **NodeStartup**: Each node (`NodeActor`) initializes a `SimpleModel`, generates synthetic data, performs local training, and sends an `UpdateModel` message to the server.
2. **ServerAggregation**: `CentralServer` collects exactly `TOTAL_NODES` updates, averages the parameters (FedAvg), updates its global model, and broadcasts the new parameters back to all nodes.
3. **Repeat**: Nodes receive the updated global model and continue local training in the next round.

## Prerequisites

- Rust (>= 1.60) and Cargo
- etcd cluster (optional, for service discovery)
- Ports opened: `5000` for server, `8001+` for nodes

## Installation

```bash
# Clone the repository
git clone https://github.com/nicktretyakov/federated_learning.git
cd federated_learning

# Build the project
cargo build --release
```

## Configuration

Set via environment variables:

| Variable        | Description                                       | Default                       |
|-----------------|---------------------------------------------------|-------------------------------|
| `RUN_AS`        | Role of process (`server` or `node`)              | `server`                     |
| `NODE_ID`       | Identifier for node (e.g., `1`, `2`, ...)         | `node1` (parsed as port offset) |
| `SERVER_ADDR`   | Central server URL (used by nodes)                | `http://127.0.0.1:5000`      |
| `TOTAL_NODES`   | Number of expected nodes for aggregation          | `2`                           |
| `ETCD_ENDPOINTS`| Comma-separated etcd endpoints for discovery      | _unset_ (discovery disabled)  |

## Usage

### Running the Central Server

```bash
export RUN_AS=server
export TOTAL_NODES=3        # e.g., expecting 3 nodes
cargo run --release
```

The server listens on `0.0.0.0:5000` by default.

### Running a Node

```bash
export RUN_AS=node
export NODE_ID=1            # node1 will use port 8002
export SERVER_ADDR=http://127.0.0.1:5000
cargo run --release
```

The node listens on port `8001 + NODE_ID` (e.g., `NODE_ID=1` → port `8002`).

### With etcd Service Discovery (Optional)

```bash
export ETCD_ENDPOINTS=http://127.0.0.1:2379
```

Nodes will register themselves under `/fedlearn/nodes/{NODE_ID}` and renew leases automatically.

## API Reference

### `/message` (POST)
- **Usage**: Exchange `NodeMessage` between nodes and server.
- **Payload**: JSON-serialized `NodeMessage` (see code in `src/messages.rs`).

### `/status` (GET)
- **Server**: Returns `{ status: "running", message: "Server is active" }`.
- **Node**: Returns `{ address: "active", status: "running" }`.

### `/api/nodes` (GET)
- Returns array of connected nodes and their statuses.

### `/api/model/params` (GET)
- Returns current model parameters and vector size.

### `/train` (POST) [Node only]
- Trigger training manually by sending `{ data: [...], labels: [...] }`.

## Code Structure

```text
src/
├── discovery.rs   # etcd-based discovery
├── messages.rs    # definitions of NodeMessage, ServerMessage, requests
├── model.rs       # SimpleModel, parameter serialization, synthetic data
├── network.rs     # HTTP handlers for server & node
├── node.rs        # NodeActor: local training, messaging
├── server.rs      # CentralServer: aggregation & broadcast
│
templates/
└── dashboard.html # Static UI for monitoring

Cargo.toml         # Dependencies and project metadata
```

## Model Details

- **SimpleModel**: 2-layer neural network (10 → 64 → 1)
  - ReLU activation in hidden layer
  - Learning rate = 0.01
  - Xavier initialization
- **Training**: Mean-squared error gradient descent for 10 epochs per round
- **Federated Averaging**: Sum parameters from each node, divide by `TOTAL_NODES`.

## Dashboard

Accessible at `http://<server-host>:5000/`. Displays:

- **Server Status**
- **Connected Nodes**
- **Global Model Parameters**

The dashboard HTML is located at `templates/dashboard.html` and is served at `/`.

## Logging & Monitoring

- Uses `env_logger` and `log` crates
- Default log level: `info`
- Enable debug: `RUST_LOG=debug cargo run`

## Contributing

Contributions are welcome! Please open issues or pull requests for enhancements, bug fixes, or new features.

---

_This documentation is autogenerated for the `federated_learning` project._

