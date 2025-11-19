import * as React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "../ui/dialog";
import { Button } from "../ui/button";
import { Loader } from "lucide-react";
import { toast } from "sonner";

interface Props {
  open: boolean;
  onClose: () => void;
  onConfirm: () => Promise<{ success: boolean; message: string }>;
  landCoverName: string;
}

export default function DeleteLandCoverModal({
  open,
  onClose,
  onConfirm,
  landCoverName,
}: Props) {
  const [loading, setLoading] = React.useState(false);

  const handleConfirm = async () => {
    try {
      setLoading(true);
      const result = await onConfirm();
      if (result.success) {
        toast.success(result.message);
        onClose();
      } else {
        toast.error(result.message);
      }
    } catch (err: any) {
      console.error("Delete failed:", err);
      toast.error(err?.message || "Failed to delete land cover.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(isOpen) => !loading && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete Land Cover</DialogTitle>
        </DialogHeader>

        <p>
          Are you sure you want to delete <b>{landCoverName}</b>?
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
            onClick={handleConfirm}
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