from api.utils.chatbot.bot import ChatBot


model = ChatBot(mode='production')
# model.train(epochs=1000)

msg = input('Message:')
print(msg)
print(model.get_predictions(msg))
