import { useState } from 'react';
import { Send } from 'lucide-react';

interface ChatInputProps {
  onSend: (message: string) => void;
  isDark: boolean;
  disabled?: boolean;
}

export function ChatInput({ onSend, isDark, disabled }: ChatInputProps) {
  const [message, setMessage] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSend(message.trim());
      setMessage('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="w-full border-t border-white/10 p-4">
      <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
        <div
          className={`relative flex items-end gap-2 rounded-xl shadow-lg ${
            isDark
              ? 'bg-[#40414F] border border-white/10'
              : 'bg-white border border-gray-300'
          }`}
        >
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message ChudGTP..."
            disabled={disabled}
            rows={1}
            className={`flex-1 resize-none bg-transparent px-4 py-3 outline-none ${
              isDark
                ? 'text-white placeholder:text-gray-400'
                : 'text-gray-900 placeholder:text-gray-500'
            } disabled:opacity-50 max-h-32`}
            style={{
              minHeight: '24px',
              height: 'auto',
            }}
          />
          <button
            type="submit"
            disabled={!message.trim() || disabled}
            className={`m-2 p-2 rounded-lg transition-colors disabled:opacity-30 ${
              isDark
                ? 'bg-white/10 hover:bg-white/20 text-white disabled:hover:bg-white/10'
                : 'bg-gray-100 hover:bg-gray-200 text-gray-900 disabled:hover:bg-gray-100'
            }`}
          >
            <Send size={18} />
          </button>
        </div>
        <p
          className={`text-xs text-center mt-2 ${
            isDark ? 'text-gray-400' : 'text-gray-500'
          }`}
        >
          ChudGTP can only make mistakes. Don't be a Chud. Check important info.
        </p>
      </form>
    </div>
  );
}