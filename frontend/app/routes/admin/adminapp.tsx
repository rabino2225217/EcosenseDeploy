import type { Route } from "./+types/adminapp";
import { Outlet } from "react-router-dom";
import { AppShell } from "../../components/admin/admin-sidebar";
import ProtectedRoute from "../protectedroute";

export function meta({}: Route.MetaArgs) {
  return [{ title: "EcoSense Admin" }];
}

export default function AppRoute() {
  return (
    <ProtectedRoute requireAdmin>
      <AppShell>
        <Outlet />
      </AppShell>
    </ProtectedRoute>
  );
}