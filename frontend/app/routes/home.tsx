import { useEffect, useState } from "react";
import { Navigate } from "react-router-dom";

export default function Home() {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setReady(true), 0);
    return () => clearTimeout(t);
  }, []);

  if (!ready) return null; 

  return <Navigate to="/login" replace />;
}