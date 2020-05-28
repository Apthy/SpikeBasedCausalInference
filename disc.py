from webhook import DiscordWebhook
webhook_urls = ['https://discordapp.com/api/webhooks/692405284265787514/J7JXbtJwIB4L6-S8U-ZIUb7oav9vCOQ5nZR-C63dxajN5FeeArBLrn8CmM80p2dO8Ikv',"https://discordapp.com/api/webhooks/692436688265281546/3hBj3f7Vk_4nuC6vL2-0JTImAr6kYy_Ko4wvKVIFg74LUpuplxvNjO6xu6VgxstRZ2EL"]

def pront(urlnum, text):
    if urlnum!=-1:
        webhook = DiscordWebhook(url=webhook_urls[urlnum], content=text)
        response = webhook.execute()