# import facebook
# import json
# from datetime import datetime
# from collections import defaultdict
# import pandas as pd


# class FaceCollector:
#     def __init__(self, access_token):
#         try:
#             self.access_token = access_token
#             self.graph = facebook.GraphAPI(access_token)
#         except Exception as e:
#             print("Loi khoi tao he thong:", e)

#     def check_token(self):
#         try:
#             me = self.graph.get_object("me", fields="id, name")
#             print("Token hop le")
#         except Exception:
#             print("Token khong hop le")
#             return False

#     def collect_data(self, limit=5):
#         try:
#             # Khai báo fields bạn cần truy vấn
#             fields = "id, message, createed_time"

#             # Lấy bài viết
#             posts = self.graph.get_object("me/feed", fields=fields, limit=limit)

#             for post in posts.get("data", []):
#                 print("--------------------------")
#                 print(post.get("message"))
#                 print("--------------------------")
#             return posts
#         except Exception:
#             pass

#     def json_to_excel(self, data, excel_file=None):
#         try:
#             posts = []
#             for post in data.get("data", []):
#                 post_data = {
#                     "id": post.get("id", "N/A"),
#                     "created_time": post.get("created_time", "N/A"),
#                     "message": post.get("message", "No message available"),
#                 }
#                 posts.append(post_data)
#             df = pd.DataFrame(posts)
#             df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce")
#             df["created_time"] = df["created_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
#             if not excel_file:
#                 timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#                 excel_file = f"facebook_posts_{timestamp}.xlsx"
#             df.to_excel(excel_file, index=False)
#             print(f"Data saved to {excel_file}")
#             return excel_file
#         except Exception as e:
#             print(f"Error exporting to Excel: {e}")
#             return None


# def main():
#     ACCESS_TOKEN = "EAAHAds4jkYoBO6cfkzrZBSxx2bXCxPwvqHykaOtJE0OQaY2QZBtfOk1uuREXxonpSi0vqJEJL9LXRf9PrreK1HpsYZAwc3gT7UfIZC3Whfh0Aeh8Ts6lOglNqoChf8ZAzwQUkCOk4PNZB4gqTGxUzit5AnOx1rLqr3TZBssQo4yRIJGYxCJSWk7Jlvd1inku8HHAYkOokX87l9tHM91ZCKIZD"
#     collector = FaceCollector(ACCESS_TOKEN)
#     if collector.check_token():
#         data = collector.collect_data(limit=2)
#         collector.json_to_excel(data)


# if __name__ == "__main__":
#     main()
import facebook
import json
from datetime import datetime
from collections import defaultdict
import pandas as pd


class FacebookCollector:
    def __init__(self, access_token):
        try:
            self.access_token = access_token
            self.graph = facebook.GraphAPI(access_token)
        except Exception as e:
            print(f"Loi khoi tao: {e}")

    def check_token_validity(self):
        try:
            me = self.graph.get_object("me", fields="id,name")
            print("Token hop le")
            return True
        except Exception:
            print("Token khong hop le")
            return False

    def collect_data(self, limit=5):
        try:
            fields = (
                "id"
                ",message"
                ",created_time"
                ",comments.limit(100).summary(true)"
                "{created_time,from{id,name},message,reactions}"
                ",reactions.limit(100).summary(true)"
                "{id,type,name}"
                ",shares"
                ",type"
            )

            # lấy sai phân
            posts = self.graph.get_object("me/feed")

            for post in posts.get("data", []):
                print("-----------------------------")
                print(post.get("message"))
                print("-----------------------------")

            return posts
        except Exception:
            pass

    def json_to_excel(self, data, excel_file=None):
        posts = []
        for post in data.get("data", []):
            post_data = {
                "id": post.get("id", ""),
                "created_time": post.get("created_time", ""),
                "message": post.get("message", ""),
            }
            posts.append(post_data)

        df = pd.DataFrame(posts)

        # chuyển đồi create_time asng định dạng datetime
        df["created_time"] = pd.to_datetime(df["created_time"])

        # format lại thời gian cho dễ đọc
        df["created_time"] = df["created_time"].dt.strftime("%Y-%m-%d")


def main():
    ACCESS_TOKEN = "EAAHAds4jkYoBO6cfkzrZBSxx2bXCxPwvqHykaOtJE0OQaY2QZBtfOk1uuREXxonpSi0vqJEJL9LXRf9PrreK1HpsYZAwc3gT7UfIZC3Whfh0Aeh8Ts6lOglNqoChf8ZAzwQUkCOk4PNZB4gqTGxUzit5AnOx1rLqr3TZBssQo4yRIJGYxCJSWk7Jlvd1inku8HHAYkOokX87l9tHM91ZCKIZD"
    collector = FacebookCollector(ACCESS_TOKEN)

    if collector.check_token_validity():
        data = collector.collect_data(limit=2)
        collector.json_to_excel(data)


if __name__ == "__main__":
    main()
