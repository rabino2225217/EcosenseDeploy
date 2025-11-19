import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "../ui/dialog";
import { Button } from "../ui/button";
import { toast } from "sonner";
import { Loader } from "lucide-react";

interface Props {
  open: boolean;
  onClose: () => void;
  onConfirm: () => Promise<boolean>; 
  userName: string;
}

export default function DeleteUserModal({ open, onClose, onConfirm, userName }: Props) {
  const [loading, setLoading] = useState(false);

  const handleDelete = async () => {
    setLoading(true);
    try {
      const success = await onConfirm();
      if (success) {
        toast.success(`User "${userName}" deleted successfully!`);
        onClose(); 
      } else {
        toast.error(`Failed to delete "${userName}". Please try again.`);
      }
    } catch (err) {
      console.error(err);
      toast.error(`An error occurred while deleting "${userName}".`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete Account</DialogTitle>
        </DialogHeader>

        <p>
          Are you sure you want to delete <b>{userName}</b>? This action cannot be undone.
        </p>

        <DialogFooter className="flex justify-end gap-2">
          <Button variant="outline" onClick={onClose} className="cursor-pointer" disabled={loading}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={handleDelete}
            className="flex items-center gap-2 cursor-pointer"
            disabled={loading}
          >
            {loading ? (
              <>
                <span>Deleting</span>
                <Loader className="h-4 w-4 animate-spin" />
              </>
            ) : (
              "Delete"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}