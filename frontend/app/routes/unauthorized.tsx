import React from "react";
import { Link } from "react-router-dom";

export default function Unauthorized() {
  const [homePath, setHomePath] = React.useState("/app");

  React.useEffect(() => {
    const checkRole = async () => {
      try {
        const res = await fetch(`${import.meta.env.VITE_API_URL}/auth/me`, {
          credentials: "include",
        });
        if (!res.ok) return;

        const user = await res.json();

        if (user.role === "Admin") {
          setHomePath("/admin");
        } else {
          setHomePath("/app");
        }
      } catch (err) {
        console.error("Failed to get role:", err);
      }
    };

    checkRole();
  }, []);

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-white px-4">
      <h1 className="text-2xl font-semibold text-gray-800 mb-2">
        Unauthorized Access
      </h1>
      <p className="text-gray-500 mb-6">
        You don’t have permission to view this page.
      </p>

      <Link
        to={homePath}
        className="px-4 py-2 rounded-md bg-[#6FA672] text-white text-sm font-medium 
                   hover:bg-[#5b8a5e] transition-colors"
      >
        Back to Home
      </Link>
    </div>
  );
}