import type { Route } from "./+types/clientapp";
import { Outlet } from "react-router-dom";
import { AppShell } from "../../components/client/app-sidebar";
import ProtectedRoute from "../protectedroute";
import { AnalysisStatusProvider } from "./analysis-status";
import { QuickGuideProvider } from "../../components/client/quick-guide-modal";
import { AnalysisStoreProvider } from "./analysis-store";

export function meta({}: Route.MetaArgs) {
  return [{ title: "EcoSense" }];
}

export default function AppRoute() {
  return (
    <ProtectedRoute requireStaff>
      <AnalysisStatusProvider>
        <AnalysisStoreProvider>
          <QuickGuideProvider>
            <AppShell>
              <Outlet />
            </AppShell>
          </QuickGuideProvider>
        </AnalysisStoreProvider>
      </AnalysisStatusProvider>
    </ProtectedRoute>
  );
}