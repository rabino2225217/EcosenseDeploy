import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "../ui/dialog";
import { Button } from "../ui/button";
import { Loader } from "lucide-react";
import { toast } from "sonner";

interface Props {
  open: boolean;
  onClose: () => void;
  onConfirm: () => Promise<{ success: boolean; message: string }>; 
  projectName: string;
}

export default function DeleteProjectModal({ open, onClose, onConfirm, projectName }: Props) {
  const [loading, setLoading] = useState(false);

  const handleDelete = async () => {
    setLoading(true);
    try {
      const { success, message } = await onConfirm();
      if (success) {
        toast.success(message);
        onClose();
      } else {
        toast.error(message);
      }
    } catch (err: any) {
      console.error(err);
      toast.error(err.message || "Something went wrong while deleting the project.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete Project</DialogTitle>
        </DialogHeader>
        <p>
          Are you sure you want to delete <b>{projectName}</b>? This action cannot be undone.
        </p>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={onClose}
            disabled={loading}
            className="cursor-pointer"
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={handleDelete}
            disabled={loading}
            className="cursor-pointer flex items-center gap-2"
          >
            {loading ? (
              <>
                Deleting
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