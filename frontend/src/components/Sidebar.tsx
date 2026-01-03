import { Plus, MessageSquare } from 'lucide-react';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  conversations: Array<{ id: string; title: string; date: string }>;
  activeConversation: string;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  isDark: boolean;
}

export function Sidebar({
  isOpen,
  onClose,
  conversations,
  activeConversation,
  onSelectConversation,
  onNewChat,
  isDark,
}: SidebarProps) {
  return (
    <>
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`fixed lg:relative top-0 left-0 h-full z-50 transition-transform duration-300 ${
          isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        } ${isDark ? 'bg-[#202123]' : 'bg-white border-r border-gray-200'} w-64 flex flex-col`}
      >
        <div className="p-3">
          <button
            onClick={onNewChat}
            className={`w-full flex items-center gap-3 px-3 py-3 rounded-lg transition-colors ${
              isDark
                ? 'bg-transparent border border-white/20 hover:bg-white/10 text-white'
                : 'bg-white border border-gray-300 hover:bg-gray-50 text-gray-900'
            }`}
          >
            <Plus size={18} />
            <span>New chat</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 pb-3">
          <div className="space-y-1">
            {conversations.map((conv) => (
              <button
                key={conv.id}
                onClick={() => onSelectConversation(conv.id)}
                className={`w-full text-left px-3 py-3 rounded-lg transition-colors flex items-center gap-3 ${
                  activeConversation === conv.id
                    ? isDark
                      ? 'bg-[#343541]'
                      : 'bg-gray-100'
                    : isDark
                    ? 'hover:bg-[#2A2B32]'
                    : 'hover:bg-gray-50'
                } ${isDark ? 'text-white' : 'text-gray-900'}`}
              >
                <MessageSquare size={16} className="flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="truncate text-sm">{conv.title}</p>
                </div>
              </button>
            ))}
          </div>
        </div>

        <div
          className={`p-3 border-t ${
            isDark ? 'border-white/20' : 'border-gray-200'
          }`}
        >
          <div
            className={`px-3 py-2 rounded-lg text-sm ${
              isDark ? 'text-gray-300' : 'text-gray-600'
            }`}
          >
            User Account
          </div>
        </div>
      </aside>
    </>
  );
}