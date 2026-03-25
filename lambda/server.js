import express from "express";
import axios from "axios";
import FormData from "form-data";
import { SessionsClient } from "@google-cloud/dialogflow-cx";

const app = express();
app.use(express.json({ limit: "20mb" }));

const BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const PROJECT_ID = process.env.PROJECT_ID;
const LOCATION = process.env.LOCATION;
const AGENT_ID = process.env.AGENT_ID;

// Default language
const DEFAULT_LANGUAGE = "en";

// Store user languages in memory
const userLanguages = {};

const dfClient = new SessionsClient({
  apiEndpoint: `${LOCATION}-dialogflow.googleapis.com`
});

function telegramApi(method) {
  return `https://api.telegram.org/bot${BOT_TOKEN}/${method}`;
}

function telegramFile(path) {
  return `https://api.telegram.org/file/bot${BOT_TOKEN}/${path}`;
}

// Root endpoint
app.get("/", (req, res) => {
  res.send("Telegram Dialogflow bot running");
});

app.post("/webhook", async (req, res) => {
  try {
    const callbackQuery = req.body?.callback_query;

    if (callbackQuery) {
      const chatId = callbackQuery.message.chat.id;
      const data = callbackQuery.data;

      // Acknowledge button click
      await axios.post(telegramApi("answerCallbackQuery"), {
        callback_query_id: callbackQuery.id
      });

      // Handle language selection
      if (data.startsWith("LANG_")) {
        const newLang = data.replace("LANG_", "");
        userLanguages[chatId] = newLang;

        await axios.post(telegramApi("sendMessage"), {
          chat_id: chatId,
          text: `Language changed to ${newLang.toUpperCase()} `
        });

        return res.sendStatus(200);
      }

      // Handle yes/no buttons
      const sessionPath = dfClient.projectLocationAgentSessionPath(
        PROJECT_ID,
        LOCATION,
        AGENT_ID,
        chatId.toString()
      );

      const langCode = userLanguages[chatId] || DEFAULT_LANGUAGE;

      const request = {
        session: sessionPath,
        queryInput: {
          text: { text: data },
          languageCode: langCode
        }
      };

      const [dfResponse] = await dfClient.detectIntent(request);

      const responseMessages = dfResponse.queryResult?.responseMessages || [];
      const textMessage = responseMessages.find(m => m.text?.text?.length);
      const replyText = textMessage?.text?.text?.[0] || "No response.";

      await axios.post(telegramApi("sendMessage"), {
        chat_id: chatId,
        text: replyText
      });

      return res.sendStatus(200);
    }

    const message = req.body?.message;
    if (!message) return res.sendStatus(200);

    const chatId = message.chat.id;

    // Typing indicator
    await axios.post(telegramApi("sendChatAction"), {
      chat_id: chatId,
      action: "typing"
    });

    const sessionPath = dfClient.projectLocationAgentSessionPath(
      PROJECT_ID,
      LOCATION,
      AGENT_ID,
      chatId.toString()
    );

    const langCode = userLanguages[chatId] || DEFAULT_LANGUAGE;

    let request;

    // Handle /change_language command
    if (message.text === "/change_language") {
      await axios.post(telegramApi("sendMessage"), {
        chat_id: chatId,
        text: "Select your preferred language:",
        reply_markup: {
          inline_keyboard: [
            [
              { text: "English", callback_data: "LANG_en" },
              { text: "Chinese", callback_data: "LANG_zh" }
            ],
            [
              { text: "Malay", callback_data: "LANG_ms" },
              { text: "Tamil", callback_data: "LANG_ta" }
            ]
          ]
        }
      });

      return res.sendStatus(200);
    }

    // TEXT MESSAGE
    if (message.text) {
      request = {
        session: sessionPath,
        queryInput: {
          text: { text: message.text },
          languageCode: langCode
        },
        outputAudioConfig: {
          audioEncoding: "OUTPUT_AUDIO_ENCODING_OGG_OPUS"
        }
      };
    }

    // VOICE MESSAGE
    if (message.voice) {
      const fileId = message.voice.file_id;

      const fileRes = await axios.get(telegramApi("getFile"), {
        params: { file_id: fileId }
      });

      const filePath = fileRes.data.result.file_path;

      const audioRes = await axios.get(telegramFile(filePath), {
        responseType: "arraybuffer"
      });

      const audioBase64 = Buffer.from(audioRes.data).toString("base64");

      request = {
        session: sessionPath,
        queryInput: {
          audio: {
            config: {
              audioEncoding: "AUDIO_ENCODING_OGG_OPUS",
              sampleRateHertz: 48000
            },
            audio: audioBase64
          },
          languageCode: langCode
        },
        outputAudioConfig: {
          audioEncoding: "OUTPUT_AUDIO_ENCODING_OGG_OPUS"
        }
      };
    }

    if (!request) {
      await axios.post(telegramApi("sendMessage"), {
        chat_id: chatId,
        text: "Please send text or voice."
      });
      return res.sendStatus(200);
    }

    // Call Dialogflow
    const [dfResponse] = await dfClient.detectIntent(request);

    const responseMessages = dfResponse.queryResult?.responseMessages || [];

    const textMessage = responseMessages.find(m => m.text?.text?.length);
    const replyText = textMessage?.text?.text?.[0] || "I heard you, but I have no response configured.";

    const payloadMessage = responseMessages.find(m => m.payload);
    const payload = payloadMessage?.payload;
    const showYesNo = payload?.telegram?.type === "yes_no";

    if (showYesNo) {
      await axios.post(telegramApi("sendMessage"), {
        chat_id: chatId,
        text: replyText,
        reply_markup: {
          inline_keyboard: [
            [
              { text: "Yes", callback_data: "YES" },
              { text: "No", callback_data: "NO" }
            ]
          ]
        }
      });
    } else {
      await axios.post(telegramApi("sendMessage"), {
        chat_id: chatId,
        text: replyText
      });
    }

    // Voice reply
    if (dfResponse.outputAudio) {
      const form = new FormData();
      form.append("chat_id", String(chatId));
      form.append("voice", Buffer.from(dfResponse.outputAudio, "base64"), {
        filename: "reply.ogg",
        contentType: "audio/ogg"
      });

      await axios.post(telegramApi("sendVoice"), form, {
        headers: form.getHeaders()
      });
    }

    res.sendStatus(200);
  } catch (err) {
    console.error("Webhook error:", err);
    res.sendStatus(200);
  }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));