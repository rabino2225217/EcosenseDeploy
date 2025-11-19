import socket from "../../services/socket";
import * as React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";
import { Button } from "../../components/ui/button";
import { Plus, Loader, Pencil, Trash2, ChevronUp, ChevronDown } from "lucide-react";
import AddLandCoverModal from "../../components/admin/add-landcover-modal";
import EditLandCoverModal from "../../components/admin/edit-landcover-modal";
import DeleteLandCoverModal from "../../components/admin/delete-landcover-modal";

const API_URL = import.meta.env.VITE_API_URL;

export default function AdminLandCover() {
  const [landCovers, setLandCovers] = React.useState<any[]>([]);
  const [loading, setLoading] = React.useState(true);

  const [showAdd, setShowAdd] = React.useState(false);
  const [showEdit, setShowEdit] = React.useState(false);
  const [showDelete, setShowDelete] = React.useState(false);
  const [selected, setSelected] = React.useState<any>(null);
  const [saving, setSaving] = React.useState(false);
  const [sortField, setSortField] = React.useState<"name" | "date_added" | null>(null);
  const [sortOrder, setSortOrder] = React.useState<"asc" | "desc" | null>(null);

  //Fetch land cover data
  React.useEffect(() => {
    const fetchLandCovers = async () => {
      try {
        const res = await fetch(`${API_URL}/layer/get`, { credentials: "include" });
        if (!res.ok) throw new Error("Failed to load land cover data");
        const data = await res.json();
        setLandCovers(data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchLandCovers();
  }, []);

  //Socket connection
  React.useEffect(() => {
    const handleUpdate = (data: any) => {
      if (data.action === "add") {
        setLandCovers((prev) => [...prev, data.layer]);
      } else if (data.action === "rename") {
        setLandCovers((prev) =>
          prev.map((l) =>
            l._id === data.layer._id ? { ...l, ...data.layer } : l
          )
        );
      } else if (data.action === "delete") {
        setLandCovers((prev) => prev.filter((l) => l._id !== data.layer._id));
      }
    };

    socket.on("landcover:update", handleUpdate);

    return () => {
      socket.off("landcover:update", handleUpdate);
    };
  }, []);

  //Sorted land covers
  const sortedLandCovers = React.useMemo(() => {
    if (!sortField) return landCovers;
    return [...landCovers].sort((a, b) => {
      let aValue: string | number = "";
      let bValue: string | number = "";

      if (sortField === "name") {
        aValue = a.name.toLowerCase();
        bValue = b.name.toLowerCase();
      } else if (sortField === "date_added") {
        aValue = a.date_added ? new Date(a.date_added).getTime() : 0;
        bValue = b.date_added ? new Date(b.date_added).getTime() : 0;
      }

      if (aValue < bValue) return sortOrder === "asc" ? -1 : 1;
      if (aValue > bValue) return sortOrder === "asc" ? 1 : -1;
      return 0;
    });
  }, [landCovers, sortField, sortOrder]);

  //Handle Add Land Cover
  const handleAdd = async (data: { name: string; type: string; file: File }): Promise<{ success: boolean; message: string }> => {
    setSaving(true);
    try {
      const formData = new FormData();
      formData.append("name", data.name);
      formData.append("type", data.type.toLowerCase());
      formData.append("file", data.file);

      const res = await fetch(`${API_URL}/admin/landcover/add`, {
        method: "POST",
        credentials: "include",
        body: formData,
      });

      const responseData = await res.json().catch(() => ({}));

      if (res.ok) {
        return { success: true, message: responseData.message || "Land cover added successfully!" };
      } else {
        return { success: false, message: responseData.error || "Failed to add land cover" };
      }
    } catch (err: any) {
      console.error("Error adding land cover:", err);
      return { success: false, message: err?.message || "Something went wrong" };
    } finally {
      setSaving(false);
    }
  };

  //Handle Edit Land Cover
  const handleEdit = async (newName: string): Promise<{ success: boolean; message: string }> => {
    if (!selected) return { success: false, message: "No land cover selected" };
    setSaving(true);
    try {
      const res = await fetch(`${API_URL}/admin/landcover/update/${selected._id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ name: newName }),
      });

      const responseData = await res.json().catch(() => ({}));

      if (res.ok) {
        setShowEdit(false);
        setSelected(null);
        return { success: true, message: responseData.message || "Land cover updated successfully!" };
      } else {
        return { success: false, message: responseData.error || "Failed to update land cover" };
      }
    } catch (err: any) {
      console.error("Error updating land cover:", err);
      return { success: false, message: err?.message || "Something went wrong" };
    } finally {
      setSaving(false);
    }
  };

  //Handle Delete Land Cover
  const handleDelete = async (): Promise<{ success: boolean; message: string }> => {
    if (!selected) return { success: false, message: "No land cover selected" };
    setSaving(true);
    try {
      const res = await fetch(`${API_URL}/admin/landcover/delete/${selected._id}`, {
        method: "DELETE",
        credentials: "include",
      });

      const responseData = await res.json().catch(() => ({}));

      if (res.ok) {
        setShowDelete(false);
        setSelected(null);
        return { success: true, message: responseData.message || "Land cover deleted successfully!" };
      } else {
        return { success: false, message: responseData.error || "Failed to delete land cover" };
      }
    } catch (err: any) {
      console.error("Error deleting land cover:", err);
      return { success: false, message: err?.message || "Something went wrong" };
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Manage Land Cover</h1>
        <Button
          className="flex items-center gap-2 cursor-pointer"
          onClick={() => setShowAdd(true)}
          disabled={saving}
        >
          <Plus className="w-4 h-4" />
          Add Land Cover
        </Button>
      </div>

      {/* Table */}
      <Card>
        <CardHeader>
          <CardTitle>Land Cover Files</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-hidden rounded-xl border border-gray-200 shadow-sm">
            <table className="w-full text-sm text-gray-700">
              <thead>
                <tr className="bg-gray-100 text-left">
                  {/* Name column */}
                  <th className="px-6 py-3 font-semibold select-none">
                    <div className="flex items-center gap-1">
                      Name
                      <div className="flex flex-col ml-1">
                        <ChevronUp
                          className={`w-3 h-3 cursor-pointer ${
                            sortField === "name" && sortOrder === "asc"
                              ? "text-emerald-500"
                              : "text-gray-400"
                          }`}
                          onClick={() => {
                            setSortField("name");
                            setSortOrder("asc");
                          }}
                        />
                        <ChevronDown
                          className={`w-3 h-3 cursor-pointer ${
                            sortField === "name" && sortOrder === "desc"
                              ? "text-emerald-500"
                              : "text-gray-400"
                          }`}
                          onClick={() => {
                            setSortField("name");
                            setSortOrder("desc");
                          }}
                        />
                      </div>
                    </div>
                  </th>

                  {/* Type column */}
                  <th className="px-6 py-3 font-semibold text-center select-none">
                    Type
                  </th>

                  {/* Date Uploaded */}
                  <th className="px-6 py-3 font-semibold text-center select-none">
                    <div className="flex items-center justify-center gap-1">
                      <span>Date Uploaded</span>
                      <div className="flex flex-col">
                        <ChevronUp
                          className={`w-3 h-3 cursor-pointer ${
                            sortField === "date_added" && sortOrder === "asc"
                              ? "text-emerald-500"
                              : "text-gray-400"
                          }`}
                          onClick={() => {
                            setSortField("date_added");
                            setSortOrder("asc");
                          }}
                        />
                        <ChevronDown
                          className={`w-3 h-3 cursor-pointer ${
                            sortField === "date_added" && sortOrder === "desc"
                              ? "text-emerald-500"
                              : "text-gray-400"
                          }`}
                          onClick={() => {
                            setSortField("date_added");
                            setSortOrder("desc");
                          }}
                        />
                      </div>
                    </div>
                  </th>

                  {/* Actions */}
                  <th className="px-6 py-3 font-semibold text-center">Actions</th>
                </tr>
              </thead>

              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={5} className="text-center p-6 text-gray-500">
                      <span className="flex items-center justify-center gap-2">
                        Loading land covers
                        <Loader className="w-5 h-5 animate-spin" />
                      </span>
                    </td>
                  </tr>
                ) : sortedLandCovers.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="text-center p-6 text-gray-500">
                      No land cover data available.
                    </td>
                  </tr>
                ) : (
                  sortedLandCovers.map((cover) => (
                    <tr
                      key={cover._id}
                      className="bg-white transition border-b border-gray-200 hover:bg-gray-50"
                    >
                      <td className="px-6 py-2">{cover.name || "N/A"}</td>
                      <td className="px-6 py-2 text-center capitalize">
                        {cover.land_type || "—"}
                      </td>
                      <td className="px-6 py-2 font-light text-center">
                        {cover.date_added
                          ? new Date(cover.date_added).toLocaleDateString("en-US", {
                              year: "numeric",
                              month: "short",
                              day: "numeric",
                            })
                          : "—"}
                      </td>
                      <td className="px-6 py-2 text-center">
                        <div className="flex items-center justify-center gap-2">
                          <Button
                            variant="ghost"
                            size="icon"
                            title="Edit"
                            className="p-2 text-gray-700 hover:bg-gray-200 cursor-pointer"
                            onClick={() => {
                              setSelected(cover);
                              setShowEdit(true);
                            }}
                            disabled={saving}
                          >
                            <Pencil className="w-4 h-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            title="Delete"
                            className="p-2 text-gray-700 hover:bg-gray-200 group cursor-pointer"
                            onClick={() => {
                              setSelected(cover);
                              setShowDelete(true);
                            }}
                            disabled={saving}
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

      {/* Modals */}
      <AddLandCoverModal
        open={showAdd}
        onClose={() => setShowAdd(false)}
        onAdd={handleAdd}
      />
      {selected && (
        <EditLandCoverModal
          open={showEdit}
          onClose={() => setShowEdit(false)}
          onSave={handleEdit}
          currentName={selected.name}
        />
      )}
      {selected && (
        <DeleteLandCoverModal
          open={showDelete}
          onClose={() => setShowDelete(false)}
          onConfirm={handleDelete}
          landCoverName={selected.name}
        />
      )}
    </div>
  );
}