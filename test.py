from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()

# Test Claude via OpenRouter
claude = ChatOpenAI(
    model="anthropic/claude-sonnet-4",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

response = claude.invoke("Say 'Hello from Claude!'")
print(response.content)

class ReviewLogger:
    """Logs review cycles to files for debugging"""
    
    def __init__(self, log_dir: str = "./review_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def log_cycle(self, segment_id: int, cycle: int, code: str, logs: str, 
                  summary: str, docs: str, success: bool):
        """
        Save a review cycle to both JSON and markdown files
        
        Args:
            segment_id: Segment identifier
            cycle: Current cycle number (1-indexed)
            code: Current code being reviewed
            logs: Execution logs
            summary: Error summary from LLM
            docs: Retrieved documentation
            success: Whether validation passed
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create segment directory
        segment_dir = os.path.join(self.log_dir, f"segment_{segment_id}")
        os.makedirs(segment_dir, exist_ok=True)
        
        # Prepare data
        cycle_data = {
            "segment_id": segment_id,
            "cycle": cycle,
            "timestamp": timestamp,
            "success": success,
            "code": code,
            "execution_logs": logs,
            "error_summary": summary,
            "retrieved_docs": docs
        }
        
        # Save as JSON
        json_path = os.path.join(segment_dir, f"cycle_{cycle}.json")
        with open(json_path, "w") as f:
            json.dump(cycle_data, f, indent=2)
        
        # Save as Markdown (more readable)
        md_path = os.path.join(segment_dir, f"cycle_{cycle}.md")
        with open(md_path, "w") as f:
            f.write(f"# Review Cycle {cycle} - Segment {segment_id}\n\n")
            f.write(f"**Timestamp:** {timestamp}\n")
            f.write(f"**Status:** {'✅ SUCCESS' if success else '❌ FAILED'}\n\n")
            
            f.write("---\n\n")
            f.write("## 📝 Code\n\n")
            f.write("```python\n")
            f.write(code)
            f.write("\n```\n\n")
            
            f.write("---\n\n")
            f.write("## 🐛 Error Summary\n\n")
            f.write(f"{summary}\n\n")
            
            f.write("---\n\n")
            f.write("## 📋 Execution Logs\n\n")
            f.write("```\n")
            f.write(logs[:2000] if len(logs) > 2000 else logs)  # Truncate very long logs
            if len(logs) > 2000:
                f.write("\n... (truncated, see JSON for full logs)\n")
            f.write("\n```\n\n")
            
            f.write("---\n\n")
            f.write("## 📚 Retrieved Documentation\n\n")
            f.write(docs if docs else "No documentation retrieved")
            f.write("\n\n")
        
        print(f"💾 Saved cycle {cycle} logs to {segment_dir}")
    
    def create_summary(self, segment_id: int, total_cycles: int, final_success: bool):
        """Create a summary file for the entire review process"""
        segment_dir = os.path.join(self.log_dir, f"segment_{segment_id}")
        summary_path = os.path.join(segment_dir, "SUMMARY.md")
        
        with open(summary_path, "w") as f:
            f.write(f"# Review Summary - Segment {segment_id}\n\n")
            f.write(f"**Total Cycles:** {total_cycles}\n")
            f.write(f"**Final Status:** {'✅ SUCCESS' if final_success else '❌ FAILED'}\n\n")
            f.write("## Cycles\n\n")
            for i in range(1, total_cycles + 1):
                f.write(f"- [Cycle {i}](cycle_{i}.md)\n")
            f.write("\n---\n\n")
            f.write(f"*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        print(f"📊 Created summary at {summary_path}")
