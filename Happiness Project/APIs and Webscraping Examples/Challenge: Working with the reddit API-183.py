## 2. Authenticating with the API ##

headers= {"Authorization": "bearer 13426216-4U1ckno9J5AiK72VRbpEeBaMSKk", "User-Agent": "Dataquest/1.0"}
response = requests.get("https://oauth.reddit.com/r/python/top", headers=headers, params = {"t":"day"})

python_top = response.json()

## 3. Getting the Most Upvoted Post ##

print(python_top)

python_top_articles= python_top["data"]["children"]
most_ups=-1
most_upvoted=-1
for article in  python_top_articles:
    if article["data"]["ups"]>most_ups:
        most_ups =article["data"]["ups"]
        most_upvoted = article["data"]["id"]
        
       
        
    
    

## 4. Getting Post Comments ##


response = requests.get("https://oauth.reddit.com/r/python/comments/4b7w9u", headers=headers)

comments=response.json()

## 5. Getting the Most Upvoted Comment ##

print(comments)

comments_list= comments[1]["data"]["children"]
most_upvotes=-1
most_upvoted_comment = -1
for comment in comments_list:
    if comment["data"]["ups"]>=most_upvotes:
        most_upvotes=comment["data"]["ups"]
        most_upvoted_comment= comment["data"]["id"]
        
        
    
    

## 6. Upvoting a Comment ##

payload={"dir": 1, "id":"d16y4ry"}
headers = {"Authorization": "bearer 13426216-4U1ckno9J5AiK72VRbpEeBaMSKk", "User-Agent": "Dataquest/1.0"}
response = requests.post("https://oauth.reddit.com/api/vote", json=payload, headers=headers)
status =response.status_code