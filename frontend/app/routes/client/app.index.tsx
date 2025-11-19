import socket from "../../services/socket";
import { Separator } from "../../components/ui/separator";
import { Search, MoreVertical, Loader, ArrowUpDown } from "lucide-react";
import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import Modal, {
  type NewProjectPayload,
} from "../../components/client/create-project-modal";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../../components/ui/dropdown-menu";
import {
  Empty,
  EmptyHeader,
  EmptyTitle,
  EmptyDescription,
} from "../../components/ui/empty";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import { toast } from "sonner";
import RenameProjectModal from "../../components/client/rename-project-modal";
import DeleteProjectModal from "../../components/client/delete-project-modal";
import { useQuickGuide } from "~/components/client/quick-guide-modal";
import React from "react";

type Project = {
  _id: string;
  name: string;
  location: string;
  description?: string;
  createdAt: string;
};

export default function AppIndex() {
  const [query, setQuery] = useState("");
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(false);
  const [renameTarget, setRenameTarget] = useState<Project | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<Project | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [sortBy, setSortBy] = useState<
    "name-asc" | "name-desc" | "date-asc" | "date-desc" | null
  >("date-desc");

  const API_URL = import.meta.env.VITE_API_URL;

  const { setCurrentSection } = useQuickGuide();

  React.useEffect(() => {
    setCurrentSection("projects");
  }, [setCurrentSection]);

  //Socket connection
  useEffect(() => {
    socket.on("project:created", (newProject) => {
      setProjects((prev) => {
        if (prev.some((p) => p._id === newProject._id)) return prev;
        return [newProject, ...prev];
      });
    });

    socket.on("project:renamed", ({ id, name }) => {
      setProjects((prev) =>
        prev.map((p) => (p._id === id ? { ...p, name } : p))
      );
    });

    socket.on("project:deleted", ({ id }) => {
      setProjects((prev) => prev.filter((p) => p._id !== id));
    });

    return () => {
      socket.off("project:created");
      socket.off("project:renamed");
      socket.off("project:deleted");
    };
  }, []);

  //Fetch projects
  useEffect(() => {
    const fetchProjects = async () => {
      setLoading(true);
      try {
        const res = await fetch(`${API_URL}/project/list`, {
          credentials: "include",
        });
        const json = await res.json();

        if (res.ok) {
          setProjects(json.data || []);
        } else {
          toast.error(json.message || "Failed to load projects.");
        }
      } catch (err) {
        console.error(err);
        toast.error("Failed to load projects.");
      } finally {
        setLoading(false);
      }
    };
    fetchProjects();
  }, []);

  //Handle create
  const handleCreate = async (payload: NewProjectPayload) => {
    try {
      const res = await fetch(`${API_URL}/project/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (res.ok) {
        return { success: true, message: data.message };
      } else {
        return { success: false, message: data.message };
      }
    } catch (err) {
      console.error(err);
      return {
        success: false,
        message: "Something went wrong while creating the project.",
      };
    }
  };

  //Handle rename
  const handleRename = async (
    id: string,
    newName: string
  ): Promise<{ success: boolean; message: string }> => {
    if (!newName.trim())
      return { success: false, message: "Project name cannot be empty." };

    try {
      const res = await fetch(`${API_URL}/project/${id}/rename`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ name: newName.trim() }),
      });

      const data = await res.json();
      const msg = data?.message;

      if (res.ok) {
        return { success: true, message: msg };
      } else {
        return { success: false, message: msg };
      }
    } catch (err: any) {
      console.error(err);
      return { success: false, message: err.message };
    }
  };

  //Handle delete
  const handleDelete = async (
    id: string
  ): Promise<{ success: boolean; message: string }> => {
    try {
      const res = await fetch(`${API_URL}/project/${id}`, {
        method: "DELETE",
        credentials: "include",
      });

      const data = await res.json();
      const msg = data?.message;

      if (res.ok) {
        return { success: true, message: msg };
      } else {
        return { success: false, message: msg };
      }
    } catch (err: any) {
      console.error(err);
      return { success: false, message: err.message };
    }
  };

  const filteredProjects = projects
    .filter((p) => p.name.toLowerCase().includes(query.toLowerCase()))
    .sort((a, b) => {
      if (!sortBy) return 0;

      switch (sortBy) {
        case "name-asc":
          return a.name.localeCompare(b.name);
        case "name-desc":
          return b.name.localeCompare(a.name);
        case "date-asc":
          return (
            new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
          );
        case "date-desc":
          return (
            new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
          );
      }
    });

  return (
    <div className="space-y-8 select-none">
      {/* Banner */}
      <div className="rounded-xl bg-[linear-gradient(90deg,#4f7d58_0%,#6FA672_40%,#1b221c_100%)] text-white shadow-lg">
        <div className="flex items-center justify-between p-6">
          <h2 className="text-xl md:text-2xl font-semibold">
            Start detection powered by EcoSense
          </h2>
          <img src="/logo-transparent.svg" alt="logo" className="h-36 w-36" />
        </div>
      </div>

      {/* Header Row */}
      <div className="flex flex-row flex-wrap items-center justify-between gap-4 w-full">
        <h3 className="text-xl sm:text-2xl font-semibold tracking-tight text-center sm:text-left flex-shrink-0">
          All projects
        </h3>

        <div className="flex flex-row flex-wrap items-center gap-2 w-full sm:w-auto">
          <div className="relative flex-grow min-w-[150px] sm:w-64">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              data-qg="search-projects"
              placeholder="Search by name or email"
              className="h-10 w-full rounded-md pl-10 pr-3 text-sm text-black dark:text-white bg-white dark:bg-[#1c1c1c] border border-gray-300 dark:border-gray-700 outline-none focus:ring-2 focus:ring-emerald-300"
            />
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
          </div>

          <div className="flex items-center gap-2 flex-shrink-0">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  aria-label="Sort"
                  data-qg="sort-projects"
                  className="inline-flex p-1 hover:scale-110 hover:text-gray-900 dark:hover:text-emerald-300 transition-transform cursor-pointer focus:outline-none"
                >
                  <ArrowUpDown className="h-5 w-5" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                align="end"
                className="w-[10rem] sm:w-[10rem] p-2 bg-white rounded-xl shadow-lg border space-y-1"
              >
                <DropdownMenuItem
                  onClick={() => setSortBy("date-desc")}
                  className={`px-3 py-2 rounded-md cursor-pointer ${
                    sortBy === "date-desc"
                      ? "bg-emerald-100 text-emerald-800 font-medium"
                      : "text-gray-700 hover:bg-gray-100"
                  }`}
                >
                  Newest → Oldest
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => setSortBy("date-asc")}
                  className={`px-3 py-2 rounded-md cursor-pointer ${
                    sortBy === "date-asc"
                      ? "bg-emerald-100 text-emerald-800 font-medium"
                      : "text-gray-700 hover:bg-gray-100"
                  }`}
                >
                  Oldest → Newest
                </DropdownMenuItem>
                <div className="my-1 h-px bg-gray-200" />
                <DropdownMenuItem
                  onClick={() => setSortBy("name-asc")}
                  className={`px-3 py-2 rounded-md cursor-pointer ${
                    sortBy === "name-asc"
                      ? "bg-emerald-100 text-emerald-800 font-medium"
                      : "text-gray-700 hover:bg-gray-100"
                  }`}
                >
                  A → Z
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => setSortBy("name-desc")}
                  className={`px-3 py-2 rounded-md cursor-pointer ${
                    sortBy === "name-desc"
                      ? "bg-emerald-100 text-emerald-800 font-medium"
                      : "text-gray-700 hover:bg-gray-100"
                  }`}
                >
                  Z → A
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            <Modal onCreate={handleCreate} />
          </div>
        </div>
      </div>

      <Separator />

      {/* Projects grid */}
      {loading ? (
        <div className="flex items-center justify-center gap-2 rounded-xl border bg-gray-100 p-10 text-gray-600">
          <span className="text-sm font-medium">Loading projects</span>
          <Loader className="w-5 h-5 animate-spin text-gray-500" />
        </div>
      ) : filteredProjects.length === 0 ? (
        <div
          className="rounded-xl border bg-gray-100 p-10 text-center text-gray-600 dark:bg-[#1c1c1c]"
          data-qg="all-projects-panel"
        >
          <Empty>
            <EmptyHeader>
              <EmptyTitle>
                <b className="font-semibold dark:text-white">
                  {query.trim() ? "No Projects Found" : "No Projects Yet"}
                </b>
              </EmptyTitle>
              <EmptyDescription>
                {query.trim() ? (
                  "We couldn’t find any projects matching your search."
                ) : (
                  <>
                    You haven't created any projects yet. Click{" "}
                    <b className="font-semibold">New</b> to create your first
                    project.
                  </>
                )}
              </EmptyDescription>
            </EmptyHeader>
          </Empty>
        </div>
      ) : (
        <div
          className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3"
          data-qg="all-projects-panel"
        >
          {filteredProjects.map((p) => (
            <Link
              key={p._id}
              to={`/app/projects/${p._id}`}
              className="
                rounded-xl border 
                bg-gradient-to-b from-white to-gray-100 
                p-5 shadow-sm 
                hover:from-gray-50 hover:to-gray-200 hover:shadow-md hover:scale-[1.01] 
                transition block
                dark:bg-gradient-to-b dark:from-[#1f1f1f] dark:to-[#111111]
                dark:border-gray-700
                dark:hover:from-[#242424] dark:hover:to-[#121212]
              "
            >
              <div className="flex justify-between items-start">
                <h4 className="text-lg font-semibold break-words line-clamp-2 leading-snug text-gray-900 dark:text-white">
                  {p.name}
                </h4>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button
                      onClick={(e) => e.preventDefault()}
                      className="rounded-full p-2 text-gray-500 hover:bg-gray-100 cursor-pointer"
                    >
                      <MoreVertical className="h-5 w-5" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-32">
                    <DropdownMenuItem
                      onClick={(e) => {
                        e.stopPropagation();
                        setRenameTarget(p);
                        setRenameValue(p.name);
                      }}
                      className="cursor-pointer"
                    >
                      Rename
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onClick={(e) => {
                        e.stopPropagation();
                        setDeleteTarget(p);
                      }}
                      className="text-red-500 cursor-pointer"
                    >
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              <p className="text-sm text-gray-600 dark:text-gray-300">
                {p.location}
              </p>
              {p.description && (
                <p className="mt-2 text-sm text-gray-800 dark:text-gray-200 line-clamp-2">
                  {p.description}
                </p>
              )}
              <p className="mt-4 text-xs text-gray-400 dark:text-gray-400">
                Created: {new Date(p.createdAt).toLocaleDateString()}
              </p>
            </Link>
          ))}
        </div>
      )}

      {/* Rename Modal */}
      {renameTarget && (
        <RenameProjectModal
          open={!!renameTarget}
          currentName={renameValue}
          onClose={() => setRenameTarget(null)}
          onRename={(newName) => handleRename(renameTarget._id, newName)}
        />
      )}

      {/* Delete Modal */}
      {deleteTarget && (
        <DeleteProjectModal
          open={!!deleteTarget}
          projectName={deleteTarget.name}
          onClose={() => setDeleteTarget(null)}
          onConfirm={() => handleDelete(deleteTarget._id)}
        />
      )}
    </div>
  );
}
