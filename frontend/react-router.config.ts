import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
  //Login routes
  index("routes/home.tsx"),
  route("login", "routes/login.tsx"),

  //Client routes
  route("client/register", "routes/client/register.tsx"),
  route("app", "routes/client/app.tsx", [
    index("routes/client/app.index.tsx"),
    route("map", "routes/client/app.map.tsx"),
    route("upload", "routes/client/app.upload.tsx"),
    route("reports", "routes/client/app.reports.tsx"),
    route("settings", "routes/client/app.settings.tsx"),
  ]),

  //Admin routes

] satisfies RouteConfig;