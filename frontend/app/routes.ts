import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
  //Routes
  index("routes/home.tsx"),
  route("login", "routes/login.tsx"),
  route("register", "routes/client/register.tsx"),
  route("unauthorized", "routes/unauthorized.tsx"),

  //Client routes
  route("app", "routes/client/clientapp.tsx", [
    //Project list
    index("routes/client/app.index.tsx"),

    //Dynamic project wrapper
    route("projects/:id", "routes/client/app.project.tsx", [
      route("analysis", "routes/client/app.analysis.tsx"),
      route("mapview", "routes/client/app.map.tsx"),
      route("summary", "routes/client/app.summary.tsx"),
    ]),

    route("settings", "routes/client/app.settings.tsx"),
  ]),

  //Admin routes
  route("admin", "routes/admin/adminapp.tsx", [
    index("routes/admin/admin.index.tsx"),
    route("projects", "routes/admin/admin.projects.tsx"),
    route("layers", "routes/admin/admin.layers.tsx"),
    route("landcover", "routes/admin/admin.landcover.tsx"),
    route("settings", "routes/admin/admin.settings.tsx")
  ]),
] satisfies RouteConfig;