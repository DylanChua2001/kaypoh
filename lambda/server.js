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
const LANGUAGE_CODE = process.env.LANGUAGE_CODE || "en";

const dfClient = new SessionsClient({
  apiEndpoint: `${LOCATION}-dialogflow.googleapis.com`
});

function telegramApi(method) {
  return `https://api.telegram.org/bot${BOT_TOKEN}/${method}`;
}

function telegramFile(path) {
  return `https://api.telegram.org/file/bot${BOT_TOKEN}/${path}`;
}

app.get("/", (req, res) => {
  res.send("Telegram Dialogflow bot running");
});

app.post("/webhook", async (req, res) => {
  try {
    const message = req.body?.message;
    if (!message) {
      return res.sendStatus(200);
    }

    const chatId = message.chat.id;

    // Show typing indicator
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

    let request;

    // TEXT MESSAGE
    if (message.text) {
      request = {
        session: sessionPath,
        queryInput: {
          text: {
            text: message.text
          },
          languageCode: LANGUAGE_CODE
        },
        outputAudioConfig: {
          audioEncoding: "OUTPUT_AUDIO_ENCODING_OGG_OPUS"
        }
      };
    }

    // VOICE MESSAGE
    if (message.voice) {
      const fileId = message.voice.file_id;

      // Get Telegram file path
      const fileRes = await axios.get(telegramApi("getFile"), {
        params: { file_id: fileId }
      });

      const filePath = fileRes.data.result.file_path;

      // Download audio
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
          languageCode: LANGUAGE_CODE
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

    // Call Dialogflow CX
    const [dfResponse] = await dfClient.detectIntent(request);

    const replyText =
      dfResponse.queryResult?.responseMessages
        ?.find(m => m.text?.text?.length)?.text?.text?.[0] ||
      "I heard you, but I have no response configured.";

    // Send TEXT reply
    await axios.post(telegramApi("sendMessage"), {
      chat_id: chatId,
      text: replyText
    });

    // Send VOICE reply if available
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

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});