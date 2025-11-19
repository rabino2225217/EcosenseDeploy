import socket from "../../services/socket";
import { useState, useEffect } from "react";
import { CheckCheck, X, Pencil, Key, Trash2, Search, Loader, ChevronUp, ChevronDown } from "lucide-react";
import AddUserModal, { type NewUserPayload } from "../../components/admin/add-user-modal";
import EditUserModal from "../../components/admin/edit-user-modal";
import ResetPasswordModal from "../../components/admin/reset-password-modal";
import DeleteUserModal from "../../components/admin/delete-user-modal";
import { Switch } from "../../components/ui/switch";
import { Button } from "../../components/ui/button";
import { toast } from "sonner";

type User = {
  _id: string;
  name: string;
  email: string;
  isActive: boolean;
  status: "pending" | "active";
};

export default function AdminIndex() {
  const [query, setQuery] = useState("");
  const [users, setUser] = useState<User[]>([]);
  const [loading, setLoading] = useState(false);

  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [isEditOpen, setEditOpen] = useState(false);
  const [isResetOpen, setResetOpen] = useState(false);
  const [isDeleteOpen, setDeleteOpen] = useState(false);
  const [sortOrder, setSortOrder] = useState<"asc" | "desc" | null>(null);
  const [tab, setTab] = useState<"ALL" | "PENDING">("ALL");

  const API_URL = import.meta.env.VITE_API_URL;

  //Socket connection
  useEffect(() => {
    socket.on("user:added", (user: User) => {
      toast.success(`New user added: ${user.name}`);
      setUser((prev) => [user, ...prev]);
    });

    socket.on("user:updated", (user: User) => {
      toast.info(`User updated: ${user.name}`);
      setUser((prev) => prev.map((u) => (u._id === user._id ? user : u)));
    });

    socket.on("user:toggled", (user: User) => {
      setUser((prev) => prev.map((u) => (u._id === user._id ? user : u)));
    });

    socket.on("user:deleted", ({ _id }: { _id: string }) => {
      toast.warning("A user was deleted.");
      setUser((prev) => prev.filter((u) => u._id.toString() !== _id.toString()));
    });

    socket.on("user:approved", (user: User) => {
      toast.success(`User approved: ${user.name}`);
      setUser((prev) => prev.map((u) => (u._id === user._id ? user : u)));
    });

    socket.on("user:rejected", ({ _id }: { _id: string }) => {
      toast.error("A user was rejected and removed.");
      setUser((prev) => prev.filter((u) => u._id !== _id));
    });

    return () => {
      socket.off("user:added");
      socket.off("user:updated");
      socket.off("user:toggled");
      socket.off("user:deleted");
      socket.off("user:approved");
      socket.off("user:rejected");
    };
  }, []);

  //Load users
  useEffect(() => {
    const loadUsers = async () => {
      setLoading(true);
      try {
        const res = await fetch(
          `${API_URL}/admin/users/get${tab === "PENDING" ? "?status=pending" : ""}`,
          { credentials: "include" }
        );
        const data = await res.json();
        setUser([...data].reverse());
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    loadUsers();
  }, [tab]);

  //Create user
  const handleCreateUser = async (payload: NewUserPayload): Promise<{ success: boolean; message: string }> => {
    try {
      const res = await fetch(`${API_URL}/admin/users/add`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        credentials: "include",
      });

      const data = await res.json(); 

      if (res.ok) {
        return { success: true, message: data.message || "User created successfully!" };
      } else {
        return { success: false, message: data.message || "Failed to create user" };
      }
    } catch (err: any) {
      console.error("Error creating user:", err);
      return { success: false, message: err?.message || "Something went wrong" };
    }
  };

  //Edit user
  const handleEditUser = async (updated: { name: string; email: string }): Promise<{ success: boolean; message: string }> => {
    if (!selectedUser) return { success: false, message: "No user selected." };

    try {
      const res = await fetch(`${API_URL}/admin/users/${selectedUser._id}/update`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updated),
        credentials: "include",
      });

      const text = await res.text(); 

      if (res.ok) {
        setUser((prev) =>
          prev.map((u) =>
            u._id === selectedUser._id ? { ...u, ...updated } : u
          )
        );
        return { success: true, message: "User updated successfully!" };
      } else {
        return { success: false, message: text || "Failed to update user." };
      }
    } catch (err) {
      console.error("Error updating user:", err);
      return { success: false, message: "An unexpected error occurred." };
    }
  };

  //Reset password
  const handleResetPassword = async (newPassword: string): Promise<boolean> => {
    if (!selectedUser) return false;
    try {
      const res = await fetch(`${API_URL}/admin/users/${selectedUser._id}/password`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password: newPassword }),
        credentials: "include",
      });

      if (res.ok) {
        return true;
      } else {
        console.error(await res.text());
        return false;
      }
    } catch (err) {
      console.error("Error resetting password:", err);
      return false;
    }
  };

  //Delete user
  const handleDeleteUser = async (id: string): Promise<boolean> => {
    try {
      const res = await fetch(`${API_URL}/admin/users/${id}/delete`, {
        method: "DELETE",
        credentials: "include",
      });

      if (res.ok) {
        setUser((prev) => prev.filter((u) => u._id !== id));
        return true;
      } else {
        console.error(await res.text());
        return false;
      }
    } catch (err) {
      console.error("Error deleting user:", err);
      return false;
    }
  };

  //Toggle active/inactive
  const handleToggleActive = async (id: string, newStatus: boolean) => {
    try {
      const res = await fetch(`${API_URL}/admin/users/${id}/active`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ isActive: newStatus }),
        credentials: "include",
      });

      if (res.ok) {
        setUser((prev) =>
          prev.map((u) =>
            u._id === id ? { ...u, isActive: newStatus } : u
          )
        );
      } else {
        console.error(await res.text());
      }
    } catch (err) {
      console.error("Error updating status:", err);
    }
  };

  //Approve user
  const handleApproveUser = async (id: string) => {
    try {
      const res = await fetch(`${API_URL}/admin/users/${id}/approve`, {
        method: "PUT",
        credentials: "include",
      });

      if (res.ok) {
        setUser((prev) =>
          prev.map((u) =>
            u._id === id ? { ...u, status: "active", isActive: true } : u
          )
        );
        toast.success("User approved successfully!");
      } else {
        const errorText = await res.text();
        console.error(errorText);
        toast.error("Failed to approve user.");
      }
    } catch (err) {
      console.error("Error approving user:", err);
      toast.error("Something went wrong while approving the user.");
    }
  };

  //Reject user
  const handleRejectUser = async (id: string) => {
    try {
      const res = await fetch(`${API_URL}/admin/users/${id}/reject`, {
        method: "DELETE",
        credentials: "include",
      });

      if (res.ok) {
        setUser((prev) => prev.filter((u) => u._id !== id));
        toast.success("User rejected successfully!");
      } else {
        const errorText = await res.text();
        console.error(errorText);
        toast.error("Failed to reject user.");
      }
    } catch (err) {
      console.error("Error rejecting user:", err);
      toast.error("Something went wrong while rejecting the user.");
    }
  };

  //Filtered users
  const filtered = users
    .filter(
      (u) =>
        [u.name, u.email].some((f) =>
          f.toLowerCase().includes(query.toLowerCase())
        ) &&
        ((tab === "ALL" && u.status === "active") || (tab === "PENDING" && u.status === "pending"))
    )
    .sort((a, b) => {
      if (!sortOrder) return 0;
      return sortOrder === "asc"
        ? a.name.localeCompare(b.name)
        : b.name.localeCompare(a.name);
    });

  return (
  <div className="p-6 space-y-6">
    {/* Header */}
    <div className="flex items-center justify-between">
      <h3 className="text-2xl font-bold text-gray-800">Manage Accounts</h3>
      <div className="flex items-center gap-3">
        <div className="relative">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by name or email"
            className="h-10 w-64 rounded-md border bg-white pl-10 pr-3 text-sm text-black outline-none focus:ring-2 focus:ring-emerald-300"
          />
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
        </div>
        <AddUserModal onCreate={handleCreateUser} />
      </div>
    </div>

    {/* Tabs */}
    <div className="flex border-b border-gray-200 mb-4">
      <button
        className={`px-4 py-2 -mb-px font-medium cursor-pointer ${
          tab === "ALL" ? "border-b-2 border-emerald-500 text-emerald-600" : "text-gray-500 hover:text-gray-700"
        }`}
        onClick={() => setTab("ALL")}
      >
        All Accounts
      </button>
      <button
        className={`px-4 py-2 -mb-px font-medium cursor-pointer ${
          tab === "PENDING" ? "border-b-2 border-emerald-500 text-emerald-600" : "text-gray-500 hover:text-gray-700"
        }`}
        onClick={() => setTab("PENDING")}
      >
        Pending Accounts
      </button>
    </div>

    {/* Table */}
    <div className="overflow-hidden rounded-xl border border-gray-200 shadow-sm">
      <table className="w-full text-sm text-gray-700">
        <thead>
          <tr className="bg-gray-100 text-left">
            <th className="px-6 py-3 font-semibold text-left cursor-pointer select-none">
              <div className="flex items-center gap-1">
                Name
                <div className="flex flex-col ml-1">
                  <ChevronUp
                    className={`w-3 h-3 ${sortOrder === "asc" ? "text-emerald-500" : "text-gray-400"}`}
                    onClick={() => setSortOrder("asc")}
                  />
                  <ChevronDown
                    className={`w-3 h-3 ${sortOrder === "desc" ? "text-emerald-500" : "text-gray-400"}`}
                    onClick={() => setSortOrder("desc")}
                  />
                </div>
              </div>
            </th>
            <th className="px-6 py-3 font-semibold text-left">Email</th>
            <th className="px-6 py-3 font-semibold text-center">Actions</th>
            {tab !== "PENDING" && (
              <th className="px-6 py-3 font-semibold text-center">Status</th>
            )}
          </tr>
        </thead>
        <tbody>
          {loading ? (
            <tr>
              <td
                colSpan={tab !== "PENDING" ? 4 : 3}
                className="text-center p-6 text-gray-500"
              >
                <span className="flex items-center justify-center gap-2">
                  Loading accounts
                  <Loader className="w-5 h-5 animate-spin" />
                </span>
              </td>
            </tr>
          ) : filtered.length === 0 ? (
            <tr>
              <td
                colSpan={tab !== "PENDING" ? 4 : 3}
                className="text-center p-6 text-gray-500"
              >
                No users found.
              </td>
            </tr>
          ) : (
            filtered.map((user) => (
              <tr
                key={user._id}
                className="bg-white border-b border-gray-200 hover:bg-gray-50 transition"
              >
                {/* Name */}
                <td className="px-6 py-3 text-left">{user.name}</td>

                {/* Email */}
                <td className="px-6 py-3 text-left">{user.email}</td>

                {/* Actions */}
                <td className="px-6 py-3 text-center">
                  <div className="flex items-center justify-center gap-2">
                    {tab === "PENDING" ? (
                      <>
                        <Button
                          title="Reject"
                          variant="ghost"
                          className="p-2 text-gray-700 hover:bg-gray-200 group cursor-pointer"
                          onClick={() => handleRejectUser(user._id)}
                        >
                          <X className="w-4 h-4 group-hover:text-red-600 transition-colors" />
                        </Button>

                        <Button
                          title="Approve"
                          variant="ghost"
                          className="p-2 text-gray-700 hover:bg-gray-200 group cursor-pointer"
                          onClick={() => handleApproveUser(user._id)}
                        >
                          <CheckCheck className="w-4 h-4 group-hover:text-emerald-600 transition-colors" />
                        </Button>
                      </>
                    ) : (
                      <>
                        <Button
                            title="Edit"
                            variant="ghost"
                            className="p-2 text-gray-700 hover:bg-gray-200 cursor-pointer"
                            onClick={() => {
                              setSelectedUser(user);
                              setEditOpen(true);
                            }}
                          >
                            <Pencil className="w-4 h-4" />
                          </Button>

                          <Button
                            title="Reset Password"
                            variant="ghost"
                            className="p-2 text-gray-700 hover:bg-gray-200 cursor-pointer"
                            onClick={() => {
                              setSelectedUser(user);
                              setResetOpen(true);
                            }}
                          >
                            <Key className="w-4 h-4" />
                          </Button>

                          <Button
                            title="Delete"
                            variant="ghost"
                            className="p-2 text-gray-700 hover:bg-gray-200 group cursor-pointer"
                            onClick={() => {
                              setSelectedUser(user);
                              setDeleteOpen(true);
                            }}
                          >
                            <Trash2 className="w-4 h-4 group-hover:text-red-600 transition-colors" />
                          </Button>
                      </>
                    )}
                  </div>
                </td>

                {/* Status */}
                {tab !== "PENDING" && (
                  <td className="px-6 py-3 text-center">
                    {user.status === "pending" ? (
                      <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-700">
                        Pending
                      </span>
                    ) : (
                      <Switch
                        checked={user.isActive}
                        onCheckedChange={(checked) =>
                          handleToggleActive(user._id, checked)
                        }
                        className="data-[state=checked]:bg-emerald-500 cursor-pointer"
                      />
                    )}
                  </td>
                )}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>

    {/* Modals */}
    {selectedUser && (
      <>
        <EditUserModal
          open={isEditOpen}
          onClose={() => setEditOpen(false)}
          onSave={handleEditUser}
          currentName={selectedUser.name}
          currentEmail={selectedUser.email}
        />
        <ResetPasswordModal
          open={isResetOpen}
          onClose={() => setResetOpen(false)}
          onReset={handleResetPassword}
          userName={selectedUser.name}
        />
        <DeleteUserModal
          open={isDeleteOpen}
          onClose={() => setDeleteOpen(false)}
          onConfirm={() => handleDeleteUser(selectedUser._id)}
          userName={selectedUser.name}
        />
      </>
    )}
  </div>
);
}