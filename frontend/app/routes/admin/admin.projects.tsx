import socket from "../../services/socket";
import { useEffect, useState } from "react";
import {
  Search,
  Loader,
  Pencil,
  Trash2,
  ChevronUp,
  ChevronDown,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "../../components/ui/card";
import { Button } from "../../components/ui/button";
import EditProjectModal from "../../components/admin/edit-project-modal";
import DeleteProjectModal from "../../components/admin/delete-project-modal";
import { toast } from "sonner";

const API_URL = import.meta.env.VITE_API_URL;

export default function AppProjects() {
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [projects, setProjects] = useState<any[]>([]);
  const [selected, setSelected] = useState<any>(null);
  const [showEdit, setShowEdit] = useState(false);
  const [showDelete, setShowDelete] = useState(false);
  const [sortOrder, setSortOrder] = useState<"asc" | "desc" | null>(null);
  const [sortField, setSortField] = useState<"name" | "createdAt" | null>(null);

  useEffect(() => {
    socket.on("project:created", (newProject) => {
      const formatted = {
        id: newProject._id,
        name: newProject.name,
        createdBy: newProject.userId?.name || "Unknown",
        createdAt: newProject.createdAt,
      };

      setProjects((prev) => {
        if (prev.some((p) => p.id === formatted.id)) return prev;
        return [formatted, ...prev];
      });
    });

    socket.on("project:renamed", ({ id, name }) => {
      setProjects((prev) =>
        prev.map((p) => (p.id === id ? { ...p, name } : p))
      );
    });

    socket.on("project:deleted", ({ id }) => {
      setProjects((prev) => prev.filter((p) => p.id !== id));
    });

    return () => {
      socket.off("project:created");
      socket.off("project:renamed");
      socket.off("project:deleted");
    };
  }, []);

  //Fetch all projects
  const fetchProjects = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/admin/projects/all`, {
        credentials: "include",
      });

      const result = await res.json();

      if (!res.ok)
        throw new Error(result.message || "Failed to fetch projects");

      const formatted = result.data.map((p: any) => ({
        id: p._id,
        name: p.name,
        createdBy: p.userId?.name || "Unknown",
        createdAt: p.createdAt,
      }));

      setProjects(formatted);
    } catch (err: any) {
      console.error("Error fetching projects:", err);
      toast.error(err.message || "Failed to load projects");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProjects();
  }, []);

  //Handle rename project
  const handleRename = async (
    newName: string
  ): Promise<{ success: boolean; message: string }> => {
    if (!selected) return { success: false, message: "No project selected" };
    try {
      const res = await fetch(
        `${API_URL}/admin/projects/${selected.id}/rename`,
        {
          method: "PUT",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: newName }),
        }
      );

      const data = await res.json();

      if (res.ok) {
        return {
          success: true,
          message: data.message || "Project renamed successfully!",
        };
      } else {
        return {
          success: false,
          message: data.message || "Failed to rename project",
        };
      }
    } catch (err: any) {
      console.error("Rename error:", err);
      return {
        success: false,
        message: err?.message || "Something went wrong",
      };
    }
  };

  //Handle delete project
  const handleDelete = async (): Promise<{
    success: boolean;
    message: string;
  }> => {
    if (!selected) return { success: false, message: "No project selected" };
    try {
      const res = await fetch(
        `${API_URL}/admin/projects/${selected.id}/delete`,
        {
          method: "DELETE",
          credentials: "include",
        }
      );

      const data = await res.json();

      if (res.ok) {
        return {
          success: true,
          message: data.message || "Project deleted successfully!",
        };
      } else {
        return {
          success: false,
          message: data.message || "Failed to delete project",
        };
      }
    } catch (err: any) {
      console.error("Delete error:", err);
      return {
        success: false,
        message: err?.message || "Something went wrong",
      };
    }
  };

  const filtered = projects.filter(
    (p) =>
      p.name.toLowerCase().includes(query.toLowerCase()) ||
      p.createdBy.toLowerCase().includes(query.toLowerCase())
  );

  const filteredSortedProjects = filtered.sort((a, b) => {
    if (!sortField) return 0;

    let aValue: string | number = "";
    let bValue: string | number = "";

    if (sortField === "name") {
      aValue = a.name.toLowerCase();
      bValue = b.name.toLowerCase();
    } else if (sortField === "createdAt") {
      aValue = new Date(a.createdAt).getTime();
      bValue = new Date(b.createdAt).getTime();
    }

    if (aValue < bValue) return sortOrder === "asc" ? -1 : 1;
    if (aValue > bValue) return sortOrder === "asc" ? 1 : -1;
    return 0;
  });

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-2xl font-bold text-gray-800">Manage Projects</h3>

        <div className="relative">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search projects..."
            className="h-10 w-64 rounded-md border bg-white pl-10 pr-3 text-sm text-black outline-none focus:ring-2 focus:ring-emerald-300"
          />
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
        </div>
      </div>

      {/* Table */}
      <Card>
        <CardHeader>
          <CardTitle>All Projects</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-hidden rounded-xl border border-gray-200 shadow-sm">
            <table className="w-full text-sm text-gray-700">
              <thead>
                <tr className="bg-gray-100 text-left">
                  <th className="px-6 py-3 font-semibold text-left select-none">
                    <div className="flex items-center gap-1">
                      Project Name
                      <div className="flex flex-col ml-1">
                        <ChevronUp
                          className={`w-3 h-3 cursor-pointer ${sortField === "name" && sortOrder === "asc" ? "text-emerald-500" : "text-gray-400"}`}
                          onClick={(e) => {
                            e.stopPropagation();
                            setSortField("name");
                            setSortOrder("asc");
                          }}
                        />
                        <ChevronDown
                          className={`w-3 h-3 cursor-pointer ${sortField === "name" && sortOrder === "desc" ? "text-emerald-500" : "text-gray-400"}`}
                          onClick={(e) => {
                            e.stopPropagation();
                            setSortField("name");
                            setSortOrder("desc");
                          }}
                        />
                      </div>
                    </div>
                  </th>

                  <th className="px-6 py-3 font-semibold">Created By</th>

                  <th className="px-6 py-3 font-semibold text-right select-none">
                    <div className="flex items-center justify-end gap-1">
                      Date Created
                      <div className="flex flex-col ml-1">
                        <ChevronUp
                          className={`w-3 h-3 cursor-pointer ${sortField === "createdAt" && sortOrder === "asc" ? "text-emerald-500" : "text-gray-400"}`}
                          onClick={(e) => {
                            e.stopPropagation();
                            setSortField("createdAt");
                            setSortOrder("asc");
                          }}
                        />
                        <ChevronDown
                          className={`w-3 h-3 cursor-pointer ${sortField === "createdAt" && sortOrder === "desc" ? "text-emerald-500" : "text-gray-400"}`}
                          onClick={(e) => {
                            e.stopPropagation();
                            setSortField("createdAt");
                            setSortOrder("desc");
                          }}
                        />
                      </div>
                    </div>
                  </th>
                  <th className="px-6 py-3 font-semibold text-center">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={4} className="text-center p-6 text-gray-500">
                      <span className="flex items-center justify-center gap-2">
                        Loading projects
                        <Loader className="w-5 h-5 animate-spin" />
                      </span>
                    </td>
                  </tr>
                ) : filtered.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="text-center p-6 text-gray-500">
                      No projects found.
                    </td>
                  </tr>
                ) : (
                  filteredSortedProjects.map((project) => (
                    <tr
                      key={project.id}
                      className="bg-white transition border-b border-gray-200 hover:bg-gray-50"
                    >
                      <td className="px-6 py-3">{project.name}</td>
                      <td className="px-6 py-3">{project.createdBy}</td>
                      <td className="px-6 py-3 font-light text-right">
                        {new Date(project.createdAt).toLocaleDateString()}
                      </td>
                      <td className="px-6 py-3 text-center">
                        <div className="flex items-center justify-center gap-2">
                          <Button
                            variant="ghost"
                            size="icon"
                            title="Edit"
                            className="p-2 text-gray-700 hover:bg-gray-200 cursor-pointer"
                            onClick={() => {
                              setSelected(project);
                              setShowEdit(true);
                            }}
                          >
                            <Pencil className="w-4 h-4" />
                          </Button>

                          <Button
                            variant="ghost"
                            size="icon"
                            title="Delete"
                            className="p-2 text-gray-700 hover:bg-gray-200 group cursor-pointer"
                            onClick={() => {
                              setSelected(project);
                              setShowDelete(true);
                            }}
                          >
                            <Trash2 className="w-4 h-4 group-hover:text-red-600 transition-colors" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Rename Modal */}
      {selected && (
        <EditProjectModal
          open={showEdit}
          onClose={() => setShowEdit(false)}
          project={selected}
          onSave={handleRename}
        />
      )}

      {/* Delete Modal */}
      {selected && (
        <DeleteProjectModal
          open={showDelete}
          onClose={() => setShowDelete(false)}
          onConfirm={handleDelete}
          projectName={selected.name}
        />
      )}
    </div>
  );
}
