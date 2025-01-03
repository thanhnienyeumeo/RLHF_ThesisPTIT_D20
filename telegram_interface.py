
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext, Application  
import json
import sys
from gpt import GPT
from configs import get_configs
from answer import reply
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = get_configs("gpt2-medium")
model_sft = GPT.from_checkpoint(
    cfg,
    r".\runs\sft\sft_sft_202411102112_final.pt")
model_ppo= GPT.from_checkpoint(
    cfg,
    r"runs\ppo_fail\ppo_ppo_202412010215_actor_step14000.pt")
# Hàm khởi động với lệnh /start trong telegram
async def start_command(update: Update, context: CallbackContext) -> None:

  await update.message.reply_text(f'Hello {update.effective_user.first_name}')

# Hàm khởi động với lệnh /help trong telegram
async def help_command(update: Update, context: CallbackContext) -> None:
  await update.message.reply_text("Bạn muốn tôi giúp gì?")

# Hàm khởi động với lệnh /ppo trong telegram
async def ppo_command(update: Update, context: CallbackContext) -> None:
  print('ppo asked')
  text = str(update.message.text).lower()[4:]
  await update.message.reply_text(reply('ppo', text, device, model_ppo))# Hàm khởi động với các câu hỏi trong file Response.py
# def handle_message(update: Update, context: CallbackContext):
#   text = str(update.message.text).lower()
#   response = R.sample_response(text)
#   update.message.reply_text(response)
async def sft_command(update: Update, context: CallbackContext) -> None:
  print('sft asked')
  text = str(update.message.text).lower()[4:]
  await update.message.reply_text(reply('sft', text, device, model_sft))

#create a function when no command is given

def handle_message(update: Update, context: CallbackContext):
  # text = str(update.message.text).lower()
  # response = reply('ppo', text, device, model_ppo)
  # update.message.reply_text(response)
  update.message.reply_text('Please use command /ppo or /sft to choose the model for the bot to reply')

def main():
  from api import API_KEY
  # updater = Updater(API_KEY)
  dp = Application.builder().token(API_KEY).build()
 
  
  # Define các lệnh mà mọi người cần
  # Param thứ nhất là nội dung câu lệnh, param thứ 2 là hàm để chạy lệnh đó
  dp.add_handler(CommandHandler('start', start_command))
  dp.add_handler(CommandHandler("help", help_command))
  dp.add_handler(CommandHandler("ppo", ppo_command))
  dp.add_handler(CommandHandler("sft", sft_command))
  # dp.add_handler(MessageHandler(Filters.text, handle_message))
  dp.run_polling() # Chạy bot



main()