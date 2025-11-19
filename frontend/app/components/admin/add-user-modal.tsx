import * as React from "react";
import { Plus, Loader } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { toast } from "sonner";

export type NewUserPayload = {
  name: string;
  email: string;
  password: string;
};

export type CreateUserResponse = {
  success: boolean;
  message: string;
};

const maxEmailLength = 100;
const maxNameLength = 50;
const minNameLength = 2;

const emailRegex = /^(?!.*\.\.)[a-zA-Z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,10}$/;
const nameRegex = /^[a-zA-ZÑñ\s'-]{2,50}$/;

export default function AddUserModal({
  onCreate,
}: {
  onCreate?: (data: NewUserPayload) => Promise<CreateUserResponse>;
}) {
  const [open, setOpen] = React.useState(false);
  const [name, setName] = React.useState("");
  const [email, setEmail] = React.useState("");
  const [password, setPassword] = React.useState("");
  const [loading, setLoading] = React.useState(false);

  React.useEffect(() => {
    if (open) {
      setName("");
      setEmail("");
      setPassword("");
    }
  }, [open]);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();

    const trimmedName = name.trim();
    const trimmedEmail = email.trim();
    const Password = password;

    if (!trimmedName || !trimmedEmail || !Password) {
      toast.error("Please fill in all fields.");
      return;
    }

    if (trimmedName.length < minNameLength || trimmedName.length > maxNameLength || !nameRegex.test(trimmedName)) {
      toast.error(`Name must be ${minNameLength}-${maxNameLength} characters and only letters, spaces, apostrophes or hyphens.`);
      return;
    }

    if (!emailRegex.test(trimmedEmail)) {
      toast.error("Please enter a valid email address.");
      return;
    }

    if (trimmedEmail.length > maxEmailLength) {
      toast.error("Email must not exceed 100 characters.");
      return;
    }

    setLoading(true);

    try {
      const response = await onCreate?.({
        name: trimmedName,
        email: trimmedEmail,
        password: Password,
      });

      if (!response) return;

      if (response.success) {
        toast.success(response.message || "User created successfully!");
        setOpen(false);
      } else {
        toast.error(response.message || "Failed to create user.");
      }
    } catch (err: any) {
      toast.error(err?.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <Button
        type="button"
        className="gap-2 rounded-md cursor-pointer"
        onClick={() => setOpen(true)}
      >
        <Plus className="h-4 w-4" />
        Add User
      </Button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center"
          role="dialog"
          aria-modal="true"
        >
          <div
            className="absolute inset-0 bg-black/40"
            onClick={() => !loading && setOpen(false)}
          />

          <div
            className="relative z-10 w-[92vw] max-w-lg rounded-2xl bg-white p-6 shadow-2xl dark:bg-neutral-900"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-xl font-semibold">Add User</h2>
            <p className="mt-1 text-sm text-gray-500 leading-tight">
              Fill out the user details below.
            </p>

            <form onSubmit={handleSubmit} className="mt-6 space-y-5">
              <div className="space-y-2">
                <Label htmlFor="user-name" className="text-sm font-medium">
                  Name
                </Label>
                <Input
                  id="user-name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Full name"
                  disabled={loading}
                  autoComplete="off"
                  className="text-base"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="user-email" className="text-sm font-medium">
                  Email
                </Label>
                <Input
                  id="user-email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="e.g. user@gmail.com"
                  disabled={loading}
                  autoComplete="off"
                  className="text-base"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="user-password" className="text-sm font-medium">
                  Password
                </Label>
                <Input
                  id="user-password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="********"
                  disabled={loading}
                  autoComplete="off"
                  className="text-base"
                />
              </div>

              <div className="mt-6 flex justify-end gap-3">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setOpen(false)}
                  disabled={loading}
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  className="bg-black hover:bg-gray-900 text-white flex items-center gap-2 cursor-pointer"
                  disabled={loading || !name.trim() || !email.trim() || !password.trim()}
                >
                  {loading ? (
                    <>
                      <span>Creating</span>
                      <Loader className="h-4 w-4 animate-spin" />
                    </>
                  ) : (
                    "Create"
                  )}
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
}