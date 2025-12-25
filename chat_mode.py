import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from providers import OpenAICompatibleProvider
from utils.logger import logger
from utils.token_counter import count_tokens

# Try to import prompt_toolkit for enhanced interactive experience
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.completion import WordCompleter
    try:
        from utils.chat_completer import ChatCompleter
    except ImportError:
        # Fallback to simple completer if custom completer not available
        ChatCompleter = None
    _PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    _PROMPT_TOOLKIT_AVAILABLE = False
    ChatCompleter = None
    try:
        logger.warning("prompt_toolkit not available, using basic input()")
    except NameError:
        pass


def print_chat_help() -> None:
    """Print help for chat meta commands."""
    help_text = """
Meta Commands (start with '/'):
  /help                 - Show this help message
  /info                 - Show current session information
  /config               - Show current configuration
  /providers            - List all available providers
  /models               - List available models for current provider
  /model [name]         - Show or switch model
  /history              - Show conversation history summary
  /save-history <file>  - Save conversation history to file
  /clear-history        - Clear conversation history (keep system prompt)
  /system-prompt [text] - Show or set system prompt
  /clear-system-prompt  - Clear system prompt

Shell Commands (start with '!'):
  !command              - Execute shell command (e.g., !ls -la)
  !!                    - Repeat last shell command
  !n                    - Execute nth command from shell history

Other:
  exit/quit/q           - Exit chat mode
  Tab                   - Auto-complete commands (if prompt_toolkit available)
  Up/Down arrows        - Browse command history (if prompt_toolkit available)
  Ctrl+R                - Search history (if prompt_toolkit available)
"""
    print(help_text)


def chat_mode(
    provider: OpenAICompatibleProvider,
    provider_name: str,
    model: Optional[str],
    temperature: Optional[float],
    initial_context: Optional[str] = None,
    prompt_template: Optional[str] = None,
    config_dict: Optional[Dict] = None,
    provider_config: Optional[Dict] = None,
) -> None:
    """
    Run interactive chat mode with meta commands support.

    Args:
        provider: Initialized provider instance
        provider_name: Provider name
        model: Model name
        temperature: Temperature parameter
        initial_context: Optional initial context (from input file)
        prompt_template: Optional prompt template
        config_dict: Full configuration dictionary
        provider_config: Provider configuration dictionary
    """
    messages: List[Dict[str, str]] = []
    system_prompt: Optional[str] = None
    current_model: Optional[str] = model
    current_provider = provider
    current_provider_name = provider_name
    current_config_dict = config_dict or {}
    current_provider_config = provider_config or {}

    # Add system prompt if exists
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    # Add initial context if provided
    if initial_context:
        if prompt_template:
            content = prompt_template.format(content=initial_context)
        else:
            content = initial_context
        messages.append({"role": "user", "content": content})
        logger.info(f"Added initial context ({len(initial_context)} characters)")
        print(
            f"Initial context: {initial_context[:100]}{'...' if len(initial_context) > 100 else ''}"
        )

    print("\n=== Interactive Chat Mode ===")
    print("Type '/help' for meta commands, 'exit'/'quit'/'q' to exit, or Ctrl+C")
    print("Type '!command' to execute shell commands (e.g., '!ls -la')")
    print("=" * 50)

    # Initialize prompt session if available
    session = None
    last_shell_command: Optional[str] = None
    shell_command_history: List[str] = []
    use_prompt_toolkit = _PROMPT_TOOLKIT_AVAILABLE

    if use_prompt_toolkit:
        try:
            # Create history file path
            history_file = Path.home() / ".ask_llm_history"
            # Fallback to current directory if home directory is not writable
            try:
                history_file.parent.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError):
                history_file = Path(".ask_llm_history")

            # Create completer
            if ChatCompleter:
                completer = ChatCompleter()
            else:
                # Fallback to word completer with meta commands
                meta_commands = [f"/{cmd}" for cmd in [
                    "help", "info", "config", "providers", "models", "model",
                    "history", "save-history", "clear-history", "system-prompt",
                    "clear-system-prompt"
                ]]
                completer = WordCompleter(meta_commands, ignore_case=True)

            # Create prompt session with history and completion
            session = PromptSession(
                history=FileHistory(str(history_file)),
                completer=completer,
                enable_history_search=True,
                complete_while_typing=True,
                complete_in_thread=False,  # Complete synchronously
            )
            logger.debug(f"Using prompt_toolkit with history file: {history_file}")
        except Exception as e:
            logger.warning(f"Failed to initialize prompt_toolkit: {e}")
            use_prompt_toolkit = False
            session = None

    def handle_meta_command(cmd: str, args: str) -> bool:
        """Handle meta commands. Returns True if command was handled."""
        nonlocal current_model, system_prompt, messages
        cmd_lower = cmd.lower()

        if cmd_lower in ["help", "h"]:
            print_chat_help()
            return True

        elif cmd_lower in ["info", "i"]:
            # Show session info
            user_msgs = sum(1 for m in messages if m["role"] == "user")
            assistant_msgs = sum(1 for m in messages if m["role"] == "assistant")
            total_tokens = sum(
                count_tokens(m["content"], current_model) for m in messages if m["role"] != "system"
            )
            print("\nSession Information:")
            print(f"  Provider: {current_provider_name}")
            print(
                f"  Model: {current_model or current_provider_config.get('api_model', 'default')}"
            )
            print(f"  Temperature: {temperature}")
            print(f"  System Prompt: {'Set' if system_prompt else 'None'}")
            print(f"  Messages: {user_msgs} user, {assistant_msgs} assistant")
            print(f"  Total Tokens (est): {total_tokens}")
            print()
            return True

        elif cmd_lower in ["config", "check-config", "c"]:
            # Show current config
            print("\nCurrent Configuration:")
            print(f"  Provider: {current_provider_name}")
            print(
                f"  Model: {current_model or current_provider_config.get('api_model', 'default')}"
            )
            print(f"  Temperature: {temperature}")
            print(f"  API Base: {current_provider_config.get('api_base', 'N/A')}")
            print(
                f"  API Key: {'Configured' if current_provider_config.get('api_key') else 'Not configured'}"
            )
            if "models" in current_provider_config:
                print(f"  Available Models: {', '.join(current_provider_config.get('models', []))}")
            print()
            return True

        elif cmd_lower in ["providers", "p"]:
            # List all providers
            providers = current_config_dict.get("providers", {})
            default_provider = current_config_dict.get("default_provider")
            print("\nAvailable Providers:")
            for name in providers.keys():
                marker = " (default)" if name == default_provider else ""
                print(f"  - {name}{marker}")
            print()
            return True

        elif cmd_lower in ["models", "m"]:
            # List available models
            models = current_provider_config.get("models", [])
            default_model = current_provider_config.get("api_model")
            if models:
                print(f"\nAvailable Models for {current_provider_name}:")
                for model_name in models:
                    marker = (
                        " (current)"
                        if model_name == current_model
                        else " (default)"
                        if model_name == default_model
                        else ""
                    )
                    print(f"  - {model_name}{marker}")
            else:
                print(f"\nNo models list configured. Default model: {default_model}")
            print()
            return True

        elif cmd_lower in ["model", "switch-model"]:
            # Show or switch model
            if not args:
                print(
                    f"\nCurrent Model: {current_model or current_provider_config.get('api_model', 'default')}"
                )
                print()
                return True

            # Switch model
            new_model = args.strip()
            available_models = current_provider_config.get("models", [])
            if available_models and new_model not in available_models:
                print(f"\n⚠️  Model '{new_model}' not in available models list.")
                print(f"Available models: {', '.join(available_models)}")
                print("Switching anyway...")

            current_model = new_model
            print(f"\n✓ Switched to model: {new_model}")
            print()
            return True

        elif cmd_lower in ["history", "hist", "h"]:
            # Show history summary
            user_msgs = [m for m in messages if m["role"] == "user"]
            assistant_msgs = [m for m in messages if m["role"] == "assistant"]
            system_msgs = [m for m in messages if m["role"] == "system"]

            print("\nConversation History:")
            print(f"  System messages: {len(system_msgs)}")
            print(f"  User messages: {len(user_msgs)}")
            print(f"  Assistant messages: {len(assistant_msgs)}")
            print(f"  Total messages: {len(messages)}")
            if messages:
                total_text = sum(len(m["content"]) for m in messages)
                print(f"  Total characters: {total_text}")
                print("\nRecent messages:")
                for i, msg in enumerate(messages[-5:], 1):
                    role = msg["role"][:4].upper()
                    content = msg["content"][:60].replace("\n", " ")
                    print(f"  {i}. [{role}] {content}...")
            print()
            return True

        elif cmd_lower in ["save-history", "save"]:
            # Save history to file
            if not args:
                print("\n⚠️  Usage: /save-history <filename>")
                print()
                return True

            filename = args.strip()
            try:
                history_data = {
                    "provider": current_provider_name,
                    "model": current_model,
                    "temperature": temperature,
                    "messages": messages,
                    "timestamp": datetime.now().isoformat(),
                }

                import json

                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(history_data, f, indent=2, ensure_ascii=False)
                print(f"\n✓ History saved to: {filename}")
            except Exception as e:
                print(f"\n✗ Failed to save history: {str(e)}")
            print()
            return True

        elif cmd_lower in ["clear-history", "clear"]:
            # Clear history but keep system prompt
            system_msgs = [m for m in messages if m["role"] == "system"]
            messages.clear()
            messages.extend(system_msgs)
            print("\n✓ Conversation history cleared (system prompt retained)")
            print()
            return True

        elif cmd_lower in ["system-prompt", "system", "prompt"]:
            # Show or set system prompt
            if not args:
                if system_prompt:
                    print("\nCurrent System Prompt:")
                    print(f"  {system_prompt}")
                else:
                    print("\nNo system prompt set.")
                print()
                return True

            # Set system prompt
            system_prompt = args
            # Remove old system messages and add new one
            messages[:] = [m for m in messages if m["role"] != "system"]
            messages.insert(0, {"role": "system", "content": system_prompt})
            print("\n✓ System prompt set")
            print()
            return True

        elif cmd_lower in ["clear-system-prompt", "clear-system"]:
            # Clear system prompt
            messages[:] = [m for m in messages if m["role"] != "system"]
            system_prompt = None
            print("\n✓ System prompt cleared")
            print()
            return True

        return False

    def execute_shell_command(cmd: str) -> bool:
        """
        Execute shell command. Returns True if command was executed.
        
        Args:
            cmd: Command string (without ! prefix)
            
        Returns:
            True if command was executed, False otherwise
        """
        nonlocal last_shell_command
        
        cmd = cmd.strip()
        if not cmd:
            return False
        
        # Handle !! (repeat last command)
        if cmd == "!!":
            if last_shell_command:
                cmd = last_shell_command
            else:
                print("No previous command to repeat.")
                return True
        
        # Handle !n (execute nth command from history)
        elif cmd.startswith("!") and cmd[1:].isdigit():
            try:
                idx = int(cmd[1:]) - 1
                if 0 <= idx < len(shell_command_history):
                    cmd = shell_command_history[idx]
                else:
                    print(f"Command history index {idx + 1} out of range.")
                    return True
            except ValueError:
                pass
        
        # Store command in history
        if cmd not in shell_command_history:
            shell_command_history.append(cmd)
        last_shell_command = cmd
        
        # Execute command
        try:
            print(f"\n[Executing: {cmd}]")
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
            )
            
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            
            if result.returncode != 0:
                print(f"\n[Command exited with code {result.returncode}]")
            else:
                print("[Command completed successfully]")
            print()
            
        except subprocess.TimeoutExpired:
            print("\n[Command timed out after 30 seconds]")
            print()
        except Exception as e:
            print(f"\n[Error executing command: {str(e)}]")
            print()
        
        return True

    try:
        while True:
            try:
                # Get user input using prompt_toolkit or standard input
                if session:
                    try:
                        user_input = session.prompt("You: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\n\nExiting chat mode...")
                        break
                else:
                    user_input = input("You: ").strip()

                # Check for exit commands
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\nExiting chat mode...")
                    break

                if not user_input:
                    continue

                # Check for shell commands (starting with !)
                if user_input.startswith("!"):
                    if execute_shell_command(user_input[1:]):
                        continue  # Shell command executed, don't send to API

                # Check for meta commands
                if user_input.startswith("/"):
                    parts = user_input[1:].split(" ", 1)
                    cmd = parts[0]
                    args = parts[1] if len(parts) > 1 else ""

                    if handle_meta_command(cmd, args):
                        continue  # Meta command handled, don't send to API

                # Regular message - send to API
                # Add user message to history
                messages.append({"role": "user", "content": user_input})

                # Prepare messages for API call (include system prompt if exists)
                api_messages = [
                    {"role": msg["role"], "content": msg["content"]} for msg in messages
                ]

                # Call API
                print("Assistant: ", end="", flush=True)
                start_time = time.time()

                response = current_provider.call(
                    messages=api_messages, temperature=temperature, model=current_model
                )

                latency = time.time() - start_time

                print(response)
                print(f"  (Latency: {latency:.2f}s)\n")

                # Add assistant response to history
                messages.append({"role": "assistant", "content": response})

            except KeyboardInterrupt:
                print("\n\nExiting chat mode...")
                break
            except Exception as e:
                logger.error(f"Error in chat: {str(e)}")
                print(f"Error: {str(e)}\n")
                # Remove the last user message if there was an error
                if messages and messages[-1]["role"] == "user":
                    messages.pop()

    except KeyboardInterrupt:
        print("\n\nExiting chat mode...")
    except Exception as e:
        logger.error(f"Chat mode error: {str(e)}")
        print(f"Error: {str(e)}")
