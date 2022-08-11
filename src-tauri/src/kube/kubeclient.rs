use std::env;
use std::path::Path;
use std::sync::mpsc::Receiver;
use futures::{StreamExt, TryStreamExt};
use kube::config::{Kubeconfig, KubeConfigOptions};
use k8s_openapi::api::core::v1::{
    ConfigMap, Namespace, Node, PersistentVolume, Pod, Secret, Service,
};
use k8s_openapi::api::apps::v1::{DaemonSet, Deployment, ReplicaSet, StatefulSet};
use kube::{
    api::{Api, ListParams, ResourceExt},
    Client, Config, Error
};
use kube::api::{LogParams, ObjectList};
use tauri::Window;
use crate::KNamespace;
use crate::kube::common::dispatch_to_frontend;
use crate::kube::metrics::{PodMetrics};
use crate::kube::models::{NodeMetrics, ResourceWithMetricsHolder};
use crate::kube::Payload;

pub struct KubeClientManager {
    cluster: String,
    kubeconfigfile: String,
    proxy_url: Option<String>,
}

impl KubeClientManager {
    pub fn clone(&self) -> Self {
        KubeClientManager {
            cluster: self.cluster.clone(),
            kubeconfigfile: self.kubeconfigfile.clone(),
            proxy_url: None
        }
    }

    pub fn initialize() -> KubeClientManager {
        KubeClientManager {
            cluster: "".to_string(),
            kubeconfigfile: "".to_string(),
            proxy_url: Some("".to_string())
        }
    }

    pub fn initialize_from(file: String, proxy_url: Option<String>) -> KubeClientManager {
        KubeClientManager {
            cluster: "".to_string(),
            kubeconfigfile: file,
            proxy_url
        }
    }

    pub fn set_cluster(&mut self, cl: &str) {
        self.cluster = cl.to_string();
    }

    pub fn set_kubeconfig_file(&mut self, file: &str) {
        self.kubeconfigfile = file.to_string();
    }

    async fn init_client(&self) -> Client {
        if self.cluster.len() > 0 {
            let kco = KubeConfigOptions {
                context: Some(self.cluster.parse().unwrap()),
                cluster: Some(self.cluster.parse().unwrap()),
                user: Some(self.cluster.parse().unwrap()),
            };
            let mut kc = Kubeconfig::read().unwrap();
            debug!("Loading custom Kubeconfig: {}", self.kubeconfigfile);
            if self.kubeconfigfile.len() > 0 {
                //TODO Check if file present
                kc = Kubeconfig::read_from(Path::new(&self.kubeconfigfile)).unwrap();
            }

            if let Some(url) = &self.proxy_url {
                if url.len() > 0 {
                    if url.starts_with("http:") {
                        std::env::set_var("HTTP_PROXY", url);
                    }else if url.starts_with("https:") {
                        std::env::set_var("HTTPS_PROXY", url);
                    }else{
                        std::env::set_var("HTTP_PROXY", url);
                    }
                }
            }

            let config = Config::from_custom_kubeconfig(kc, &kco).await;
            Client::try_from(config.unwrap()).unwrap()
        } else {
            if self.kubeconfigfile.len() > 0 {
                //TODO Check if file present
                let kc = Kubeconfig::read_from(Path::new(&self.kubeconfigfile)).unwrap();
            }
            Client::try_default().await.unwrap()
        }
    }

    pub fn get_all_ns(&self, window: &Window, cmd: &str, custom_ns_list: Vec<KNamespace>) {
        self._get_all_ns(window, cmd, custom_ns_list);
    }

    #[tokio::main]
    async fn _get_all_ns(
        &self,
        window: &Window,
        cmd: &str,
        custom_ns_list: Vec<KNamespace>
    ) -> Result<Vec<KNamespace>, Box<dyn std::error::Error>> {
        let mut kns_list: Vec<KNamespace> = Vec::new();
        let client = self.init_client().await;
        let ns_request: Api<Namespace> = Api::all(client);
        let ns_list = ns_request.list(&ListParams::default()).await?;
        for ns in ns_list {
            debug!("{:?}", ns);
            kns_list.push(KNamespace {
                creation_ts: None,
                name: ns.name_any()
            })
        }
        for cns in custom_ns_list {
            kns_list.push(cns);
        }
        let json = serde_json::to_string(&kns_list).unwrap();
        dispatch_to_frontend(window, cmd, json);
        Ok(kns_list)
    }

    pub fn get_resource_with_metrics(
        &self,
        window: &Window,
        namespace: String,
        kind: String,
        cmd: String,
    ) {
        let window_copy1 = window.clone();
        if kind == "pod" {
            self._get_pods_with_metrics(&window_copy1, &namespace, &cmd);
        } else if kind == "node" {
            self._get_nodes_with_metrics(&window_copy1, &cmd);
        } else if kind == "deployment" {
            self._get_deployments_with_metrics(&window_copy1, &namespace, &cmd);
        }
    }

    #[tokio::main]
    async fn _get_pods_with_metrics(
        &self,
        window: &Window,
        namespace: &String,
        cmd: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let client = self.init_client().await;
        let metrics_client = client.clone();
        let kube_request: Api<Pod> = Api::namespaced(client, namespace);

        let lp = ListParams::default();
        let pods: ObjectList<Pod> = kube_request.list(&lp).await?;

        let m_kube_request: Api<PodMetrics> = Api::namespaced(metrics_client, namespace);
        let lp = ListParams::default();
        let metrics = m_kube_request.list(&lp).await?;
        let json = ResourceWithMetricsHolder {
            resource: serde_json::to_string(&pods).unwrap(),
            metrics: serde_json::to_string(&metrics).unwrap(),
            usage: None,
            metrics2: None
        };
        dispatch_to_frontend(window, cmd, serde_json::to_string(&json).unwrap());
        Ok(())
    }

    #[tokio::main]
    async fn _get_nodes_with_metrics(
        &self,
        window: &Window,
        cmd: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let client = self.init_client().await;

        let metrics_client = client.clone();
        let kube_request: Api<Node> = Api::all(client);

        let lp = ListParams::default();
        let nodes: ObjectList<Node> = kube_request.list(&lp).await?;

        let m_kube_request: Api<NodeMetrics> = Api::all(metrics_client);
        let lp = ListParams::default();
        let metrics = m_kube_request.list(&lp).await?;
        let json = ResourceWithMetricsHolder {
            resource: serde_json::to_string(&nodes).unwrap(),
            metrics: serde_json::to_string(&metrics).unwrap(),
            usage: None,
            metrics2: None
        };
        let result = serde_json::to_string(&json).unwrap();
        dispatch_to_frontend(window, cmd, result);
        Ok(())
    }

    #[tokio::main]
    async fn _get_deployments_with_metrics(
        &self,
        window: &Window,
        namespace: &String,
        cmd: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let client = self.init_client().await;
        let metrics_client = client.clone();
        let pod_metrics_client = client.clone();
        let pod_client = client.clone();
        let kube_request: Api<Deployment> = Api::namespaced(client, namespace);

        let lp = ListParams::default();
        let deployments: ObjectList<Deployment> = kube_request.list(&lp).await?;

        let m_kube_request: Api<PodMetrics> = Api::namespaced(metrics_client, namespace);
        let lp = ListParams::default();
        let metrics = m_kube_request.list(&lp).await?;

        let p_kube_request: Api<Pod> = Api::namespaced(pod_client, namespace);
        let lp = ListParams::default();
        let pods = p_kube_request.list(&lp).await?;

        let mp_kube_request: Api<PodMetrics> = Api::namespaced(pod_metrics_client, namespace);
        let lp = ListParams::default();
        let pod_metrics = mp_kube_request.list(&lp).await?;

        let json = ResourceWithMetricsHolder {
            resource: serde_json::to_string(&deployments).unwrap(),
            metrics: serde_json::to_string(&metrics).unwrap(),
            usage: Some(serde_json::to_string(&pods).unwrap()),
            metrics2: Some(serde_json::to_string(&pod_metrics).unwrap())

        };
        dispatch_to_frontend(window, cmd, serde_json::to_string(&json).unwrap());
        Ok(())
    }

    #[tokio::main]
    pub async fn get_pods_for_deployment(
        &self,
        ns: &String,
        deployment: &str,
    ) -> Result<Vec<Pod>, Error> {
        self._get_pods_for_deployment(ns, deployment).await
    }

    async fn _get_pods_for_deployment(&self,
        ns: &String,
        deployment: &str,
    ) -> Result<Vec<Pod>, Error> {
        let client = self.init_client().await;
        let deploy_request: Api<Deployment> = Api::namespaced(client, ns);
        let d = deploy_request.get(deployment).await?;
        let mut pods_for_deployments: Vec<Pod> = Vec::new();
        if let Some(spec) = d.spec {
            if let Some(match_labels) = spec.selector.match_labels {
                let pclient = self.init_client().await;
                let pod_request: Api<Pod> = Api::namespaced(pclient, ns);
                debug!("Spec:: {:?}", match_labels);
                for lbl in match_labels {
                    match lbl {
                        (key, value) => {
                            debug!("Label selector:: {:?}", value);
                            let label = format!("{}={}", key.as_str(), value.as_str());
                            let lp = ListParams::default().labels(label.as_str());
                            let pods = pod_request.list(&lp).await?;
                            debug!("Total pods found {:?}", pods.items.len());
                            for pod in pods {
                                pods_for_deployments.push(pod);
                            }
                        }
                    }
                }
            }
        }
        return Ok(pods_for_deployments);
    }

    pub fn tail_logs_for_pod(
        &self,
        window: Window,
        pod: &str,
        ns: &str,
        rx: &Receiver<String>,
    ) {
        self._tail_logs_for_pod(window,  pod, ns, rx);
    }

    #[tokio::main]
    async fn _tail_logs_for_pod(
        &self,
        window: Window,
        pod: &str,
        ns: &str,
        rx: &Receiver<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        info!("Fetching logs for {:?}", pod);
        let client = self.init_client().await;
        let pods: Api<Pod> = Api::namespaced(client, ns);
        let mut logs = pods
            .log_stream(
                &pod,
                &LogParams {
                    follow: true,
                    tail_lines: Some(1),
                    ..LogParams::default()
                },
            )
            .await?
            .boxed();

        debug!("Spawning task");
        while let Some(line) = logs.try_next().await? {
            let line_str = String::from_utf8_lossy(&line);
            debug!("{:?}", line_str);
            let stopword = rx.try_recv().unwrap_or("ERR".to_string());
            if stopword != "ERR" {
                debug!("Work is done: {:?}", stopword);
                break;
            }
            window
                .emit(
                    "dashboard::logs",
                    Payload {
                        message: line_str.to_string(),
                        metadata: String::from(pod),
                    },
                )
                .unwrap();
        }
        debug!("Finished spawned task");
        Ok(())
    }

}