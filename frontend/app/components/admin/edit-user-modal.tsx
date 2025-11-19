import { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "../ui/dialog";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { toast } from "sonner";
import { Loader } from "lucide-react";

interface Props {
  open: boolean;
  onClose: () => void;
  onSave: (updated: { name: string; email: string }) => Promise<{ success: boolean; message: string }>;
  currentName: string;
  currentEmail: string;
}

const maxEmailLength = 50;
const emailRegex = /^(?!.*\.\.)[a-zA-Z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,10}$/;
const nameRegex = /^[a-zA-ZÑñ\s'-]{2,50}$/;

export default function EditUserModal({ open, onClose, onSave, currentName, currentEmail }: Props) {
  const [name, setName] = useState(currentName);
  const [email, setEmail] = useState(currentEmail);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (open) {
      setName(currentName);
      setEmail(currentEmail);
    }
  }, [open, currentName, currentEmail]);

  const handleSave = async () => {
    const trimmedName = name.trim();
    const trimmedEmail = email.trim();

    if (!trimmedName || !trimmedEmail) {
      toast.error("Please fill out all fields.");
      return;
    }

    if (!nameRegex.test(trimmedName)) {
      toast.error("Name must be 2-50 characters and only letters, spaces, apostrophes, or hyphens.");
      return;
    }

    if (!emailRegex.test(trimmedEmail)) {
      toast.error("Please enter a valid email address.");
      return;
    }

    if (trimmedEmail.length > maxEmailLength) {
      toast.error("Email must not exceed 50 characters.");
      return;
    }

    setLoading(true);
    try {
      const result = await onSave({ name: trimmedName, email: trimmedEmail });

      if (result.success) {
        toast.success(result.message || "User updated successfully!");
        onClose();
      } else {
        toast.error(result.message || "Failed to update user.");
      }
    } catch (err) {
      console.error(err);
      toast.error("An error occurred while updating the user.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit User</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
            <Input
              placeholder="Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={loading}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
            <Input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={loading}
            />
          </div>
        </div>

        <DialogFooter className="flex justify-end gap-2">
          <Button variant="outline" onClick={onClose} disabled={loading}>
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            disabled={loading || !name.trim() || !email.trim()}
            className="bg-black hover:bg-gray-900 text-white flex items-center gap-2 cursor-pointer"
          >
            {loading ? (
              <>
                <span>Saving</span>
                <Loader className="h-4 w-4 animate-spin" />
              </>
            ) : (
              "Save"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}