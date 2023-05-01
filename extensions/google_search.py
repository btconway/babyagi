import requests
import re
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
from newspaper import Article, ArticleException, Config
from requests_html import HTMLSession
import time
import random


# Different user agents setups
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:80.0) Gecko/20100101 Firefox/80.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 13_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:73.0) Gecko/20100101 Firefox/73.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.129 Safari/537.36"
]

# Google custom search API for query, return snippets and links
def search(query, api_key, search_engine_id, num_results, start_index):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": search_engine_id,
        "q": query,
        "num": num_results,
        "start": start_index
    }
    response = requests.get(url, params=params, timeout=10)

    if response.status_code == 200:
        try:
            json_data = response.json()
            if "items" in json_data:
                return json_data["items"], 0
            else:
                print("No items found in the response.")
                return [], 0
        except ValueError as e:
            print(f"Error while parsing JSON data: {e}")
            return [], 0
    else:
        print(f"Error: {response.status_code}")
        return [], response.status_code


# Get snippets for current page
def search_google(query, api_key, search_engine_id, num_results, start_index):
    results, error_code = search(query, api_key, search_engine_id, num_results, start_index)
    return [result['snippet'] for result in results], [result['link'] for result in results], error_code


# Collect all snippets (for multiple top pages)
def get_snippets(topic, api_key, search_engine_id, num_results, num_pages):
    all_snippets = []
    links = []
    for page in range(1, num_pages * num_results, num_results):
        snippets, link, error_code = search_google(topic, api_key, search_engine_id, num_results, start_index=page)
        all_snippets.extend(snippets)
        links.extend(link)

    return all_snippets, links, error_code


# Extract webpage content with Newspaper3k
def extract_with_3k(url):
    try:
        if url.lower().endswith(".pdf"):
            response = requests.get(url)
            response.raise_for_status()

            with BytesIO(response.content) as pdf_data:
                reader = PdfFileReader(pdf_data)
                content = " ".join([reader.getPage(i).extract_text() for i in range(reader.getNumPages())])

        else:
            config = Config()
            config.browser_user_agent = random.choice(USER_AGENTS)
            config.request_timeout = 5
            session = HTMLSession()

            response = session.get(url)
            response.html.render(timeout=config.request_timeout)
            html_content = response.html.html

            article = Article(url, config=config)
            article.set_html(html_content)
            article.parse()
            content = article.text.replace('\t', ' ').replace('\n', ' ').strip()

        return content[:1500]

    except ArticleException as ae:
        print(f"Error while extracting text from HTML (newspaper3k): {str(ae)}")
        return ""

    except RequestException as re:
        print(f"Error while making the request to the URL (newspaper3k): {str(re)}")
        return ""

    except Exception as e:
        print(f"Unknown error while extracting text from HTML (newspaper3k): {str(e)}")
        return ""



# Extract webpage content with BeautifulSoup
def extract_with_bs4(url):
    headers = {
        "User-Agent": random.choice(USER_AGENTS)
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'head', 'link', 'meta', 'noscript']):
                tag.decompose()

            main_content_areas = soup.find_all(['main', 'article', 'section', 'div'])
            if main_content_areas:
                main_content = max(main_content_areas, key=lambda x: len(x.text))
                content_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
                content = ' '.join([tag.text.strip() for tag in main_content.find_all(content_tags)])
            else:
                content = ' '.join([tag.text.strip() for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])

            content = re.sub(r'\t', ' ', content)
            content = re.sub(r'\s+', ' ', content)
            return content[:1500]
        else:
            print(f"Error while extracting text from HTML (bs4): {response.status_code}")
            return ""

    except Exception as e:
        print(f"Unknown error while extracting text from HTML (bs4): {str(e)}")
        return ""


# Extract webpage content with lxml
def extract_with_lxml(url):
    try:
        config = Config()
        config.browser_user_agent = random.choice(USER_AGENTS)
        config.request_timeout = 5
        session = HTMLSession()

        response = session.get(url)
        response.html.render(timeout=config.request_timeout)
        html_content = response.html.html

        tree = html.fromstring(html_content)
        paragraphs = tree.cssselect('p, h1, h2, h3, h4, h5, h6')
        content = ' '.join([para.text_content() for para in paragraphs if para.text_content()])
        content = content.replace('\t', ' ').replace('\n', ' ').strip()

        return content[:1500]

    except ArticleException as ae:
        print(f"Error while extracting text from HTML (lxml): {str(ae)}")
        return ""

    except RequestException as re:
        print(f"Error while making the request to the URL (lxml): {str(re)}")
        return ""

    except Exception as e:
        print(f"Unknown error while extracting text from HTML (lxml): {str(e)}")
        return ""


# API: Get top list results and top page content
# maximum for num_results=10, num_extracts=3
def get_toplist(query, api_key, search_engine_id, num_results, num_pages, num_extracts):
    snippets, links, error_code = get_snippets(query, api_key, search_engine_id, num_results, num_pages)
    if error_code == 429:
        snippets, links = browser_search(query)
        
    webpages = []
    attempts = 0
    while snippets == [] and attempts < 2:
        attempts += 1
        print("Google blocked the request. Trying again...")
        time.sleep(3)
        snippets, links = get_snippets(query, api_key, search_engine_id, num_results, num_pages)
        
    if links:
        for i in range(0, num_extracts):
            time.sleep(3)
            content = extract_with_3k(links[i])
            attempts = 0
            while content == "" and attempts < 2:
                attempts += 1
                content = extract_with_3k(links[i])
            if content == "":
                time.sleep(3)
                content = extract_with_bs4(links[i])
                attempts = 0
                while content == "" and attempts < 2:
                    attempts += 1
                    content = extract_with_bs4(links[i])
            webpages.append(content)
    else:
        snippets = ["", "", ""]
        links = ["", "", ""]
        webpages = ["", "", ""]

    #print(f"Top results: {snippets}\n")
    #print("Top page: " + webpage)
    return snippets, webpages, links 
