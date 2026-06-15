# Minimal image so Glama / any host can build & run the MCP server.
FROM python:3.12-slim
RUN pip install --no-cache-dir verifiable-memory-mcp
ENV VMEM_STATE=/data
VOLUME ["/data"]
ENTRYPOINT ["verifiable-memory"]
