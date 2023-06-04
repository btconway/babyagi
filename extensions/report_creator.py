import os

ENABLE_REPORT_EXTENSION = os.getenv("ENABLE_REPORT_EXTENSION", "false").lower() == "true"
REPORT_FILE = os.getenv("REPORT_FILE", "report.txt")
OBJECTIVE = os.getenv("OBJECTIVE", "")
ACTION = os.getenv("ACTION", "")

title = str("")
content = str("")
report = []


# Write to summary file
def write_report(file: str, text: str, mode: chr):
    try:
        with open(file, mode) as f:
            f.write(text)
    except Exception as e:
        print(f"Error writing to report file: {e}")      


# Read from summary file
def read_report(file: str):
    try:
        with open(file, 'r') as f:
            text = f.read()
            return text
    except Exception as e:
        return f"Error reading from report file: {e}"
    

# Get report structure
def get_report():
    return report


# Check text needs to be written to report file
def check_report(result: str):
    if ("Task List" or "Task list" and "task list") not in result[0:50]:
        try:
            # Detect code blocks (continuous appending to file)
            if "```" in result:
                print('Code block tag ``` detected...')
                file_name = REPORT_FILE.split(".")[0] + "_code.txt"
                if result.count("```") > 1:
                    block_counter = int(result.count("```")/2)
                else:
                    block_counter = result.count("```")
                for i in range(block_counter):
                    write_report(file_name, "\n```" + result.split("```")[i+1], 'a')

                write_report(file_name, "\n", 'a')
                print(f'{block_counter} code blocks written to file: {file_name}')

            # Detect text blocks (update of file)
            elif "###" in result and "'###'" not in result:
                print('Tag ### detected in result...')
                file_name = REPORT_FILE.split(".")[0] + "_text.txt"
                parts = result.split("###")
                block_counter = int(0)
                write_flag = False
                for part in parts:
                    if "###" in result[0:30] and len(part) > 50 and not part.startswith("As an AI assistant") and "Task:" not in part[0:20] and part.find("\n") != -1:
                        line = part.split("\n")
                        title = line[0].strip()
                        if len(title) < 50:
                            write_flag = True
                            block_counter += 1
                            print(f'Tag ### detected in part: {block_counter}')
                            content = part.split(title)[1].strip()
                            content = content.replace("\n", " ")

                            report_result = {
                                "title": title,
                                "content": content
                            }
                            print(f'\nDetected title: {title}')
                            print(f'Content of block:\n{content}\n')

                            # Check new report entry for append, insert or ignore
                            insert_flag = True
                            if len(content) > (50 - len(title)):
                                insert_flag = False
                                for r in report:
                                    print(f'Checking title: {r["title"]}')
                                    if title in r["title"]:
                                        insert_flag = True
                                        if content not in r["content"]:
                                            title = r["title"]
                                            content = r["content"] + " " + content
                                            report_result = {
                                                "title": title,
                                                "content": content
                                            }

                                            report.remove(r)
                                            report.append(report_result)
                                            print(f'New content for title: {r["title"]}\n')
                                        else:
                                            print(f'No update for title: {r["title"]}\n')
                                        break

                            if not insert_flag:
                                report.append(report_result)
                                print(f'Added new title to database: {report_result["title"]}\n')

                # Write report to file
                if len(report) > 0 and write_flag:
                    print(f'\nWriting report to file: {file_name}\n')
                    input = ""
                    counter = int(0)
                    for r in report:
                        counter += 1
                        print(f'Write title: {r["title"]}')
                        print(f'Write content: {r["content"]}\n')
                        input += f'### {r["title"]}\n{r["content"]}\n\n'

                    with open(file_name, 'w') as f:
                        f.write(f'# In this file BabyAGI stores a report, as configured in .env file under ENABLE_REPORT_EXTENSION.\n\n')
                        f.write(f'OBJECTIVE: {OBJECTIVE}')
                        if ENABLE_REPORT_EXTENSION:
                            f.write(f'ACTION: {ACTION}')
                        f.write('\n---------------------------\n')
                        f.write(input)
                    print(f'Report file updated with {counter} blocks, written to file: {file_name}')

        except Exception as e:
            print(f"Error: Checking results for adding to report file failed with {e}")
    
    return report


# Check if report file exists
def check_report_file(file_path: str, text: str):
    res = False
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if OBJECTIVE in lines[2]:
            print(f"Objective is unchanged, use existing report file: {file_path}")

            # Setup report structure from existing file
            if "_text.txt" in file_path:
                start_index = lines.index("---------------------------\n") + 1
                parts = ""
                for i in range (start_index, len(lines)):
                    parts += lines[i].strip() + "\n"
                parts = parts.split("###")
                print()
                for i in range (1, len(parts)):
                    title = parts[i].split("\n")[0].strip()
                    content = parts[i].split("\n")[1].strip()
                    report_result = {
                        "title": title,
                        "content": content
                    }
                    report.append(report_result)
                    print(f'Reading report part {i}: {report_result["title"]}')
                    print(f'{report_result["content"]}\n')
                print(f"Report has been initialized with {len(report)} parts.")
        res = True

    if not res:
        with open(file_path, 'w') as f:
            print(f"{text} report file objective has changed or file does not exist, new file: {file_path}")
            f.write(f'# In this file BabyAGI stores a report, as configured in .env file under ENABLE_REPORT_EXTENSION.\n\n')
            f.write(f'OBJECTIVE: {OBJECTIVE}')
            if ENABLE_REPORT_EXTENSION:
                f.write(f'\nACTION: {ACTION}')
            f.write('\n---------------------------\n')

    return report
