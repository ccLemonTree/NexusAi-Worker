# GitHub Actions 镜像构建

推送任意分支会在自托管 Runner 上测试、构建并推送：

- `aisfh5.com:9091/nexusai/nexusai-inference:vYYYYMMDDHHMMSS`

仓库 Secrets：`REGISTRY_USERNAME`、`REGISTRY_PASSWORD`。

Runner 工作目录：`/ai/chen/github-actions-runner/nexusai-worker/_work`；它需要 `self-hosted`、`linux`、`docker` 标签及 Docker 访问权限。
