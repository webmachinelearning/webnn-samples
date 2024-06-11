import {defineConfig} from 'vite';
import monacoEditorPlugin from 'vite-plugin-monaco-editor';
import basicSsl from '@vitejs/plugin-basic-ssl';

export default defineConfig({
  build: {
    target: 'esnext',
  },
  plugins: [
    basicSsl(),
    monacoEditorPlugin({languageWorkers: [
      'editorWorkerService',
    ]}),
  ],
});
