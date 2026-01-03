interface MessageProps {
  role: 'user' | 'assistant';
  content: string;
  isDark: boolean;
}

export function Message({ role, content, isDark }: MessageProps) {
  const isUser = role === 'user';
  // USE LOCAL IMAGE FROM PUBLIC FOLDER
  const botAvatarUrl = "/bot-avatar.png";

  return (
    <div
      className={`w-full ${
        isUser
          ? isDark
            ? 'bg-[#343541]'
            : 'bg-gray-50'
          : isDark
          ? 'bg-[#444654]'
          : 'bg-white'
      }`}
    >
      <div className="max-w-3xl mx-auto px-4 py-6 flex gap-6">
        <div className="flex-shrink-0">
          {isUser ? (
            <div
              className={`w-8 h-8 rounded-sm flex items-center justify-center ${
                isDark ? 'bg-[#5D5E72]' : 'bg-gray-300'
              }`}
            >
              <span className="text-white">U</span>
            </div>
          ) : (
            <img 
              src={botAvatarUrl} 
              alt="ChudGTP" 
              className="w-8 h-8 rounded-full object-cover"
            />
          )}
        </div>

        <div className="flex-1 min-w-0">
          <div
            className={`prose prose-sm max-w-none ${
              isDark ? 'text-gray-100' : 'text-gray-900'
            }`}
          >
            {content.split('\n').map((line, i) => (
              <p key={i} className="mb-2 last:mb-0">
                {line || '\u00A0'}
              </p>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}