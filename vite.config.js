import {defineConfig} from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

export default defineConfig({
  build: {
    target: 'esnext',
  },
  plugins: [basicSsl()],
});
