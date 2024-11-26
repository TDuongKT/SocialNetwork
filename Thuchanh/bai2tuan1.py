from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import os
from datetime import datetime
import getpass
from openpyxl import Workbook

import pandas as pd


class FacebookGroupScraper:
    def __init__(self):
        print("\n===FACEBOOK GROUP MEMBER SCRAPER===")
        self.get_config()
        self.setup_driver()

    def get_config(self):
        try:
            print("NHAP THONG TIN DANG NHAP:")
            self.email = input("Email/Username: ").strip()
            self.password = getpass.getpass("Password: ")

            print("\n Nhap ID group facebook: ")
            self.group_id = input("GroupID: ").strip()

            print("\nSo lan scroll de load")
            self.scroll_count = int(input("So lan scroll(mac dinh la 5)") or "5")

        except Exception as e:
            print(f"loi cau hinh: {e}")

    def setup_driver(self):
        try:
            self.driver = webdriver.Chrome()
            self.driver.maximize_window()
        except Exception as e:
            print(f"loi khoi tao trinh duyet: {e}")

    def login(self):
        try:
            self.driver.get("https://www.facebook.com")
            email_input = self.driver.find_element(By.ID, "email")
            email_input.send_keys(self.email)

            pass_input = self.driver.find_element(By.ID, "pass")
            pass_input.send_keys(self.password)

            login_botton = self.driver.find_element(By.NAME, "login")
            login_botton.click()

            time.sleep(10)
            print("dang nhap thanh cong")
            return True
        except Exception as e:
            print(f"loi dang nhap: {e}")
            return False

    def get_group_members(self):
        try:
            self.driver.get(f"https://www.facebook.com/groups/{self.group_id}/members")
            time.sleep(5)
            members = set()
            for i in range(self.scroll_count):
                self.driver.execute_script(
                    "window.scrollTo(0, document.body.scrollHeight);"
                )
                time.sleep(3)
                print(f"Scroll lan {i+1}/{self.scroll_count}")

                # thu thap thong tin thanh vien moi lan scroll
                user_elements = self.driver.find_elements(
                    By.CSS_SELECTOR, "a[href*='/user/']"
                )
                print(len(user_elements))
                for user in user_elements:
                    try:
                        href = user.get_attribute("href")
                        if "/user/" in href:
                            user_id = href.split("/user/")[1]
                            name = user.text
                            members.add((user_id, name))
                            print(user_id, " - ", name)
                    except Exception as e:
                        pass
            return list(members)
        except Exception as e:
            pass

    def export_to_excel(self, members, output_file="FacebookGroupMembers.xlsx"):
        try:
            # Kiểm tra xem members là list chứa tuple (user_id, name)
            if isinstance(members, list):
                # Tạo DataFrame từ danh sách
                df = pd.DataFrame(members, columns=["User ID", "Name"])
            else:
                raise ValueError("Dữ liệu phải là list các tuple hoặc dictionary.")

            # Ghi DataFrame ra file Excel
            df.to_excel(output_file, index=False)
            print(f"Dữ liệu đã được xuất ra file: {output_file}")
        except Exception as e:
            print(f"Lỗi khi xuất file Excel: {e}")


def main():
    scraper = None
    try:
        scraper = FacebookGroupScraper()
        if scraper.login():
            members = scraper.get_group_members()
            if members:
                scraper.export_to_excel(members)
            else:
                print("Không tìm thấy thành viên nào.")
        time.sleep(10)
    except Exception as e:
        pass


if __name__ == "__main__":
    main()