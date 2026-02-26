import CheckIcon from "@mui/icons-material/Check";

interface CheckboxProps {
  label: string;
  checked?: boolean;
  onChange?: (checked: boolean) => void;
}

export default function Checkbox({ label, checked = true, onChange }: CheckboxProps) {
  return (
    <div
      className="flex items-center justify-end gap-2 cursor-pointer select-none"
      onClick={() => onChange?.(!checked)}
    >
      <div
        className={`w-5 h-5 flex items-center justify-center rounded text-sm text-white transition-colors ${
          checked ? "bg-blue-600" : "bg-gray-600 border border-gray-400"
        }`}
      >
        {checked && <CheckIcon color="inherit" fontSize="inherit" />}
      </div>
      <p className="text-white text-lg">{label}</p>
    </div>
  );
}
