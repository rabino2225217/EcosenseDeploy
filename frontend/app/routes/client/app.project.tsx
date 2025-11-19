import { Navigate, Outlet, useParams, useLocation } from "react-router-dom";

export default function AppProject() {
  const { id } = useParams();
  const location = useLocation();

  if (location.pathname === `/app/projects/${id}`) {
    return <Navigate to="analysis" replace />;
  }

  return <Outlet />;
}