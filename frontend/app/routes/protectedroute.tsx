import socket from "../services/socket";
import React from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Loader } from "lucide-react";

export default function ProtectedRoute({
  children,
  requireAdmin = false,
  requireStaff = false,
}: {
  children: React.ReactNode;
  requireAdmin?: boolean;
  requireStaff?: boolean;
}) {
  const navigate = useNavigate();
  const location = useLocation();
  const [checked, setChecked] = React.useState(false);
  const [authorized, setAuthorized] = React.useState(false);

  React.useEffect(() => {
    let active = true;

    const verifyAuth = async () => {
      try {
        const res = await fetch(`${import.meta.env.VITE_API_URL}/auth/me`, {
          credentials: "include",
        });

        if (!res.ok) {
          if (active) navigate("/login", { replace: true });
          return;
        }

        const user = await res.json();

        if (requireAdmin && user.role !== "Admin") {
          if (active) navigate("/unauthorized", { replace: true });
          return;
        }

        if (requireStaff && !["DENR staff", "Admin"].includes(user.role)) {
          if (active) navigate("/unauthorized", { replace: true });
          return;
        }

        if (active && !socket.connected) {
          socket.connect();
        }

        if (active) setAuthorized(true);
      } catch (err) {
        if (active) navigate("/login", { replace: true });
      } finally {
        if (active) setChecked(true);
      }
    };

    verifyAuth();

    return () => {
      active = false;
    };
  }, [navigate, location.pathname, requireAdmin, requireStaff]);

  if (!checked) {
    return (
      <div className="flex items-center justify-center h-screen text-gray-500 space-x-2">
        <span className="text-sm font-medium">Loading EcoSense</span>
        <Loader className="h-5 w-5 animate-spin text-[#6FA672]" />
      </div>
    );
  }

  return authorized ? <>{children}</> : null;
}