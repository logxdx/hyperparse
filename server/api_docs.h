#pragma once
// The server's own API documentation, from one endpoint table: the endpoint list
// in --help, GET / (JSON index) and GET /docs (full text docs). Keep it in step
// with routes.h and the handlers.

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "server_config.h"

namespace hp::server {

struct Endpoint {
  const char *method;
  const char *path;
  const char *summary;
};

const std::vector<Endpoint> &endpoints();

// "Endpoints:" section for --help.
std::string endpoints_help();

// GET /: service name, where the docs are, and the endpoint list.
nlohmann::json api_index();

// GET /docs: the full API reference; base_url ("http://host:port") goes into
// the curl examples, cfg supplies the limits.
std::string api_docs(const ServerConfig &cfg, const std::string &base_url);

}  // namespace hp::server
